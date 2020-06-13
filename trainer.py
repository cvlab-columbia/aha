import hashlib
import itertools
import json
import os
import time
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from pandas import DataFrame
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from tqdm import trange, tqdm

import utils
from misc.confmat import pretty_plot_confusion_matrix
from models import gt_index
from samplers import UniformClipSamplerMulti
from utils import save_checkpoint, AverageMeter


class Trainer:
    def __init__(self, model, optim, train_loader, val_loader, args, device, tok=None, epoch=0, writer=None,
                 global_step=0,
                 best_loss=float("inf")):
        if args.fp16:
            try:
                from apex import amp
                global amp
                amp = amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

        self.model = model
        self.optim = optim
        self.train_loader = train_loader
        self.val_loader = val_loader
        # self.tok = tok or BertTokenizer.from_pretrained(
        #     'bert-base-uncased')  # Vocab not actually used, just utility functions
        self.epoch = epoch
        self.writer = writer
        self.best_loss = best_loss
        self.global_step = global_step
        self.args = args
        self.device = device

        self.clip_loss_weights = None
        self.clip_loss_probs = None

    def train(self):
        if self.args.weighted_clip_loss or self.args.undersample_clip_loss:
            args_to_hash = ['train_data_fraction', 'use_pseudo_oops_gt', 'use_oops_only', 'max_trajectory_size',
                'min_trajectory_size', 'load_entire_video', 'max_trajectories_per_video', 'frame_rate',
                'frames_per_clip', 'step_between_clips', 'num_clips_between_clips', 'p_no_sep_seq', 'min_xfmr_seq_len',
                'p_swap_seq_order', 'use_kinetics_only']
            values_to_hash = [vars(self.args)[arg] for arg in args_to_hash]
            args_hash = hashlib.sha224(str(values_to_hash).encode('utf-8')).hexdigest()
            fn = os.path.join(os.path.dirname(self.args.checkpoint_dir),
                              f'clip_loss_weights_{args_hash}_maxnorm_.json')
            try:
                clip_loss_weights = torch.load(fn, map_location=self.device)
            except FileNotFoundError:
                clip_loss_weights = self.get_clip_loss_weights()
                torch.save(clip_loss_weights, fn)
            clip_loss_weights = clip_loss_weights.to(self.device)
            if self.args.undersample_clip_loss:
                clip_loss_probs = 1 / clip_loss_weights  # c / max(c)
                clip_loss_probs = min(clip_loss_probs) / clip_loss_probs  # sampling probs
                self.clip_loss_probs = clip_loss_probs * self.args.clip_sample_prob_scale
            else:
                self.clip_loss_weights = clip_loss_weights
            if self.args.local_rank <= 0:
                print(f'Loss {"weights" if self.args.weighted_clip_loss else "sampling probabilities"}'
                      f' are: {(clip_loss_weights if self.args.weighted_clip_loss else clip_loss_probs).tolist()}')
        for epoch in trange(self.epoch, self.args.epochs, desc='Training model'):
            if self.args.local_rank != -1:
                self.train_loader.sampler.set_epoch(epoch)

            self.run_epoch(epoch)

            val_loss = self.run_epoch(epoch, train=False)

            is_best = val_loss < self.best_loss
            self.best_loss = min(val_loss, self.best_loss)

            if self.args.local_rank <= 0:
                print('Saving checkpoint')
                save_checkpoint(self.model, self.optim, epoch, val_loss, self.args.checkpoint_dir, self.global_step,
                                is_best, amp=amp, args=self.args)

    def get_clip_loss_weights(self):
        torch.cuda.synchronize()

        # Initialize self.meters
        c = defaultdict(int)

        self.model.train()

        self.train_loader.dataset.oops.get_label_only = True
        try:
            self.train_loader.dataset.kinetics.get_label_only = True
        except:
            pass

        with torch.set_grad_enabled(False), tqdm(self.train_loader,
                                                 desc=f'Getting loss weights',
                                                 disable=self.args.local_rank > 0) as t:
            for batch_idx, (videos, clip_ys, lens, *_) in enumerate(t):
                for y in clip_ys.tolist():
                    c[y] += 1
                t.set_postfix(**{str(k): v for k, v in c.items()})

        print(self.args.local_rank, c)

        c = torch.Tensor([c[0], c[1], c[2]]).to(self.device)

        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(c)
            if self.args.local_rank <= 0:
                print('gathered', c)
        else:
            pass

        c = max(c) / c
        # z = sum(c)
        # c /= (z / len(c))

        self.train_loader.dataset.oops.get_label_only = False
        try:
            self.train_loader.dataset.kinetics.get_label_only = False
        except:
            pass

        return c

    def run_epoch(self, epoch, train=True):
        torch.cuda.synchronize()

        # Initialize self.meters
        avg_batch_time = AverageMeter()
        avg_data_time = AverageMeter()
        self.meters = defaultdict(AverageMeter)

        if train:
            self.model.train()
        else:
            self.model.eval()

        end = time.time()

        with torch.set_grad_enabled(train), tqdm(self.train_loader if train else self.val_loader,
                                                 desc=f'Training epoch {epoch}' if train else f'Validating {f"epoch {epoch}" if epoch else ""}',
                                                 disable=self.args.local_rank > 0) as t:
            for batch_idx, (videos, clip_ys, lens, ys_, *_) in enumerate(t):
                # Measure data loading time
                avg_data_time.update(time.time() - end)
                end = time.time()

                videos = videos.to(self.device)
                lens = lens.to(self.device)
                clip_ys = clip_ys.to(self.device)

                clip_loss, nsp_loss, clip_acc, nsp_acc, clip_y_preds, nsp_y_preds, clip_ys, nsp_ys, lens, self.cnn_reprs, \
                    self.xfmr_reprs, raw_xfmr_reprs, idxs_, video_ts_ = self.model(
                    videos, lens, clip_ys, clip_loss_weights=self.clip_loss_weights if train else None,
                    clip_loss_sample_probs=self.clip_loss_probs if train else None)

                clip_preds = clip_y_preds.argmax(dim=1)[clip_ys != -1].tolist()
                clip_ys_ = clip_ys[clip_ys != -1].tolist()

                loss = clip_loss + self.args.nsp_loss_lambda * nsp_loss

                if loss != loss:
                    print('NaN loss, ignoring')
                    continue

                self.meters['clip_loss'].update(clip_loss.item(), sum(clip_ys != -1).item())
                self.meters['nsp_loss'].update(nsp_loss.item(), sum(nsp_ys != -1).item())
                self.meters['clip_acc'].update(clip_acc, sum(clip_ys != -1).item())
                self.meters['nsp_acc'].update(nsp_acc, sum(nsp_ys != -1).item())
                if len(clip_ys_) > 0:
                    self.meters['conf_mat'].update(confusion_matrix(clip_ys_, clip_preds, labels=range(3)), 1)
                    self.meters['conf_mat'].avg = self.meters['conf_mat'].sum

                if train:
                    loss = loss / self.args.grad_accumulate_steps

                    if self.args.fp16:
                        with amp.scale_loss(loss, self.optim) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss.backward()

                    if batch_idx and batch_idx % self.args.grad_accumulate_steps == 0:
                        self.optim.step()
                        self.optim.zero_grad()

                # Measure elapsed time
                avg_batch_time.update(time.time() - end)
                end = time.time()

                t.set_postfix(
                    t_data=avg_data_time.avg,
                    t_batch=avg_batch_time.avg,
                    **{
                        k: v.avg for k, v in self.meters.items() if 'loss' in k or 'acc' in k
                    }
                )

                if train:
                    if self.global_step % self.args.log_freq == 0 and self.writer and not self.args.debug and batch_idx:
                        self.writer.add_scalars('train/loss',
                                                {
                                                    k: v.avg for k, v in self.meters.items() if 'loss' in k
                                                },
                                                self.global_step * self.args.batch_size * self.args.step_n_gpus)
                        self.writer.add_scalars('train/acc',
                                                {
                                                    k: v.avg for k, v in self.meters.items() if 'acc' in k
                                                },
                                                self.global_step * self.args.batch_size * self.args.step_n_gpus)
                        df_cm = DataFrame(self.meters['conf_mat'].sum, index=['before', 'at', 'after'],
                                          columns=['before', 'at', 'after'])
                        conf_mat = np.array(Image.open(pretty_plot_confusion_matrix(df_cm)))[:, :, :-1]
                        self.writer.add_image('train/conf_mat', conf_mat,
                                              self.global_step * self.args.batch_size * self.args.step_n_gpus,
                                              dataformats='HWC')

                    self.global_step += 1

        if train:
            return self.meters['loss'].avg
        if not train:
            gathered_metrics = {}
            for k, v in self.meters.items():
                if 'loss' in k or 'acc' in k:
                    gathered_metrics[k] = utils.gather_score(v.avg, v.count)
                elif 'mat' in k:
                    v_ = torch.from_numpy(v.avg).to(self.device)
                    torch.distributed.all_reduce(v_)
                    gathered_metrics[k] = v_.cpu().numpy()

            if self.args.local_rank <= 0:
                print(gathered_metrics)

            if epoch is not None and self.writer is not None:
                self.writer.add_scalars('val/loss', {k: v for k, v in gathered_metrics.items() if 'loss' in k}, epoch)
                self.writer.add_scalars('val/acc', {k: v for k, v in gathered_metrics.items() if 'acc' in k}, epoch)
                df_cm = DataFrame(gathered_metrics['conf_mat'], index=['before', 'at', 'after'],
                                  columns=['before', 'at', 'after'])
                conf_mat = np.array(Image.open(pretty_plot_confusion_matrix(df_cm)))[:, :, :-1]
                self.writer.add_image('val/conf_mat', conf_mat, epoch, dataformats='HWC')

            return (gathered_metrics['clip_loss'] + self.args.nsp_loss_lambda * gathered_metrics['nsp_loss'])

    def autocorrect(self, epoch=None):
        torch.cuda.synchronize()

        self.args.test_name += '_' + str(epoch)

        # Initialize self.meters
        avg_batch_time = AverageMeter()
        avg_data_time = AverageMeter()
        self.meters = defaultdict(AverageMeter)

        self.model.eval()

        end = time.time()

        self.cnn_reprs = []
        self.xfmr_reprs = []
        self.autocorrect_reprs = []
        self.autocorrect_xfmr_reprs = []
        self.autocorrect_reprs_perstep = []
        self.autocorrect_xfmr_reprs_perstep = []
        self.fn_ts = []

        results = []

        with torch.set_grad_enabled(True), tqdm(self.val_loader,
                                                desc=f'Running autocorrect',
                                                disable=self.args.local_rank > 0) as t:
            for batch_idx, (videos, clip_ys, lens, ys, idxs, video_ts) in enumerate(t):
                # Measure data loading time
                avg_data_time.update(time.time() - end)
                end = time.time()

                videos = videos.to(self.device)
                lens = lens.to(self.device)
                clip_ys = clip_ys.to(self.device)

                auto_idx, autocorrect_stats, clip_y_preds, clip_ys_, orig_clip_y_preds, _ = self.autocorrect_batch(
                    clip_ys,
                    idxs,
                    lens,
                    video_ts,
                    videos)

                # if self.args.local_rank <= 0:
                #     import ipdb
                #     ipdb.set_trace()

                self.meters['nsp_correction_freq'].update(max(autocorrect_stats['nsp_loss_correction']))
                self.meters['n_iters'].update(auto_idx)
                self.meters['delta_nsp_loss'].update(
                    autocorrect_stats['nsp_loss'][-1] - autocorrect_stats['nsp_loss'][0])
                self.meters['clip_acc'].update(autocorrect_stats['clip_acc'][-1])
                for n in [5, 10, 15, 20, 25]:
                    self.meters[f'success_at_{n}'].update(100 * (auto_idx <= n))
                self.meters[f'reached_100_clip_acc'].update(100 * (autocorrect_stats['clip_acc'][-1] == 100))
                self.meters['delta_preoops_acc'].update(
                    autocorrect_stats['preoops_clip_acc'][-1] - autocorrect_stats['preoops_clip_acc'][0])

                # self.meters['clip_loss'].update(clip_loss.item(), sum(clip_ys != -1).item())
                # # self.meters['nsp_loss'].update(nsp_loss.item(), sum(nsp_ys != -1).item())
                # self.meters['clip_acc'].update(clip_acc, sum(clip_ys != -1).item())
                # # self.meters['nsp_acc'].update(nsp_acc, sum(nsp_ys != -1).item())
                # # Measure elapsed time
                avg_batch_time.update(time.time() - end)
                end = time.time()

                if self.args.save_test_results:
                    # if self.args.local_rank <= 0:
                    #     import ipdb
                    #     ipdb.set_trace()
                    clip_ys_ = clip_ys_.chunk(len(ys))
                    video_ts = video_ts.split(lens.tolist())
                    clip_y_preds = clip_y_preds.chunk(len(ys))
                    orig_clip_y_preds = orig_clip_y_preds.chunk(len(ys))
                    for idx, ts, preds, gts, preds_orig in zip(idxs, video_ts, clip_y_preds, clip_ys_,
                                                               orig_clip_y_preds):
                        preds_ = preds.argmax(dim=1)[gts != -1].tolist()
                        pred_probs = preds[gts != -1].tolist()
                        preds_orig_ = preds_orig.argmax(dim=1)[gts != -1].tolist()
                        pred_probs_orig = preds_orig[gts != -1].tolist()
                        gts = gts[gts != -1].tolist()
                        fn = self.val_loader.dataset.get_filename_by_idx(idx)
                        ts = ts.tolist()
                        results.append({
                            'fn': fn,
                            'ts': ts,
                            'preds_final': preds_,
                            'preds_orig': preds_orig_,
                            'pred_probs': pred_probs,
                            'pred_probs_orig': pred_probs_orig,
                            'gts': gts,
                            **{k: v.val for k, v in self.meters.items()}
                        })

                t.set_postfix(
                    t_data=avg_data_time.avg,
                    t_batch=avg_batch_time.avg,
                    **{
                        k: v.avg for k, v in self.meters.items()
                    }
                )

        with torch.no_grad():
            if 'retrieval' in self.args.test_names:
                self.cnn_reprs = torch.cat(self.cnn_reprs)
                self.cnn_reprs = self.cnn_reprs / self.cnn_reprs.norm(p=2, dim=-1).unsqueeze(-1)
                self.xfmr_reprs = torch.cat(self.xfmr_reprs)
                self.xfmr_reprs = self.xfmr_reprs / self.xfmr_reprs.norm(p=2, dim=-1).unsqueeze(-1)
                self.autocorrect_reprs = torch.stack(self.autocorrect_reprs)
                self.autocorrect_reprs = self.autocorrect_reprs / self.autocorrect_reprs.norm(p=2, dim=-1).unsqueeze(-1)
                self.autocorrect_xfmr_reprs = torch.cat(self.autocorrect_xfmr_reprs)
                self.autocorrect_xfmr_reprs = self.autocorrect_xfmr_reprs / self.autocorrect_xfmr_reprs.norm(p=2,
                                                                                                             dim=-1).unsqueeze(
                    -1)
                # cnn_nn_scores, cnn_nn_idxs = torch.triu(self.cnn_reprs @ self.cnn_reprs.t()).topk(k=5)
                # xfmr_nn_scores, xfmr_nn_idxs = torch.triu(self.xfmr_reprs @ self.xfmr_reprs.t()).topk(k=5)
                # autocorrect_nn_scores, autocorrect_nn_idxs = torch.triu(
                #     self.autocorrect_reprs @ (self.cnn_reprs if self.args.autocorrect_cnn_output else self.xfmr_reprs).t()).topk(k=5)
                os.makedirs(os.path.join(self.args.results_dir, self.args.test_name), exist_ok=True)
                # torch.save({
                #     'cnn_nn_scores': cnn_nn_scores, 'cnn_nn_idxs': cnn_nn_idxs,
                #     'xfmr_nn_scores': xfmr_nn_scores, 'xfmr_nn_idxs': xfmr_nn_idxs,
                #     'autocorrect_nn_scores': autocorrect_nn_scores, 'autocorrect_nn_idxs': autocorrect_nn_idxs,
                #     'self.fn_ts': self.fn_ts
                # }, os.path.join(self.args.results_dir, self.args.test_name, f'retrievals_{self.args.local_rank}.pt'))
                torch.save({
                    'cnn_reprs': self.cnn_reprs,
                    'xfmr_reprs': self.xfmr_reprs,
                    'autocorrect_reprs': self.autocorrect_reprs,
                    'autocorrect_xfmr_reprs': self.autocorrect_xfmr_reprs,
                    'autocorrect_xfmr_reprs_perstep': self.autocorrect_xfmr_reprs_perstep,
                    ''
                    'autocorrect_cnn': self.args.autocorrect_cnn_output,
                    'fn_ts': self.fn_ts
                }, os.path.join(self.args.results_dir, self.args.test_name, f'retrievals_{self.args.local_rank}.pt'))

        gathered_metrics = {}
        for k, v in self.meters.items():
            gathered_metrics[k] = utils.gather_score(v.avg, v.count)

        if self.args.local_rank <= 0:
            print(gathered_metrics)

        if self.args.save_test_results:
            os.makedirs(os.path.join(self.args.results_dir, self.args.test_name), exist_ok=True)
            with open(os.path.join(self.args.results_dir, self.args.test_name,
                                   f'results_{self.args.local_rank}.json'),
                      'w') as f:
                json.dump(results, f)

    def autocorrect_batch(self, clip_ys, idxs, lens, video_ts, videos, svo_mode=False):
        clip_loss, nsp_loss, clip_acc, nsp_acc, clip_y_preds, nsp_y_preds, clip_ys_, nsp_ys, lens_, \
            self.cnn_reprs_, xfmr_reprs_, raw_xfmr_reprs, *_ = self.model(videos, lens, clip_ys, override_no_svo=True)
        orig_clip_y_preds = clip_y_preds
        with torch.no_grad():
            if not svo_mode and 'retrieval' in self.args.test_names:
                self.cnn_reprs.extend(map(lambda x: x.detach().cpu(), self.cnn_reprs_))
                self.xfmr_reprs.extend(map(lambda x: x.detach().cpu(), xfmr_reprs_))
                video_ts_ = video_ts.detach().cpu().split(lens.tolist())
                self.fn_ts.extend(zip(map(self.val_loader.dataset.get_filename_by_idx, idxs), video_ts_))
        # if self.args.local_rank <= 0:
        #     import ipdb
        #     ipdb.set_trace()
        orig_preoops_acc = 100 * (((clip_y_preds.argmax(dim=1)[clip_ys_ == 0]) == 0).sum().item() / (
                sum(clip_ys_ == 0).item() or 1))
        autocorrect_stats = defaultdict(list)
        orig_nsp_loss = nsp_loss.item()
        if not svo_mode:
            self.meters['orig_preoops_acc'].update(orig_preoops_acc, sum(clip_ys_ == 0).item())
        autocorrect_stats['preoops_clip_acc'].append(orig_preoops_acc)
        autocorrect_stats['nsp_loss'].append(orig_nsp_loss)
        y_autocorrect = torch.zeros_like(clip_ys) - 1
        y_autocorrect[clip_ys > 0] = 0
        y_autocorrect_ = torch.zeros_like(clip_ys_) - 1
        y_autocorrect_[clip_ys_ > 0] = 0
        x_autocorrect = torch.cat(self.cnn_reprs_) if self.args.autocorrect_cnn_output else raw_xfmr_reprs
        x_autocorrect = x_autocorrect.detach().clone().requires_grad_()
        x_ac_per_step = []
        x_ac_xfmr_per_step= []
        delta_x = torch.zeros_like(x_autocorrect)
        autocorrect_dict = {}
        if not self.args.autocorrect_cnn_output:
            autocorrect_dict['_xfmr_tuple'] = [None, lens_, y_autocorrect_, nsp_ys]
        for auto_idx in range(self.args.max_autocorrect_iters):
            x_ac_per_step.append((x_autocorrect + delta_x).detach().cpu())

            if self.args.autocorrect_cnn_output:
                autocorrect_dict['_clip_embs'] = x_autocorrect + delta_x
            else:
                autocorrect_dict['_xfmr_tuple'][0] = x_autocorrect + delta_x

            clip_loss, nsp_loss, clip_acc, nsp_acc, clip_y_preds, *_ = self.model(None, lens.tolist(),
                                                                                  y_autocorrect,
                                                                                  **autocorrect_dict,
                                                                                  override_no_svo=True)

            x_ac_xfmr_per_step.append(_[-3].cpu()[clip_ys_.view(len(_[-3]), -1) != -1].split(lens.tolist()))

            autocorrect_loss = clip_loss + self.args.nsp_loss_lambda * max(0, nsp_loss - orig_nsp_loss)
            preoops_clip_acc = (100 * (clip_y_preds.argmax(dim=1)[clip_ys_ == 0] == 0).sum().item() / (
                    (clip_ys_ == 0).sum().item() or 1))

            autocorrect_stats['nsp_loss_correction'].append(100 if nsp_loss > orig_nsp_loss else 0)
            autocorrect_stats['clip_loss'].append(clip_loss.item())
            autocorrect_stats['nsp_loss'].append(nsp_loss.item())
            autocorrect_stats['clip_acc'].append(clip_acc)
            autocorrect_stats['nsp_acc'].append(nsp_acc)
            autocorrect_stats['preoops_clip_acc'].append(preoops_clip_acc)

            if clip_acc == 100:
                break

            (grad,) = torch.autograd.grad(autocorrect_loss, x_autocorrect)

            if self.args.autocorrect_cnn_output:
                delta_x[y_autocorrect == 0] -= self.args.autocorrect_alpha * \
                                               torch.sign(grad)[y_autocorrect == 0]
            else:
                delta_x[:, y_autocorrect_ == 0] -= self.args.autocorrect_alpha * \
                                                   torch.sign(grad)[:, y_autocorrect_ == 0]

            delta_x.clamp_(-self.args.autocorrect_eps, self.args.autocorrect_eps)

            # x_autocorrect.grad.data.zero_()
        x_autocorrect = x_autocorrect + delta_x
        with torch.no_grad():
            if not svo_mode and 'retrieval' in self.args.test_names:
                if not self.args.autocorrect_cnn_output:
                    x_autocorrect = x_autocorrect[clip_ys_.unsqueeze(0) != -1]
                self.autocorrect_reprs.extend(x_autocorrect.detach().cpu())
                self.autocorrect_reprs_perstep.append(x_ac_per_step)
                self.autocorrect_xfmr_reprs_perstep.append(x_ac_xfmr_per_step)
                self.autocorrect_xfmr_reprs.extend(
                    _[-3].cpu()[clip_ys_.view(len(_[-3]), -1) != -1].split(lens.tolist()))
        if svo_mode:
            autocorrect_dict['_xfmr_tuple'][2] = clip_ys_
        return auto_idx, autocorrect_stats, clip_y_preds, clip_ys_, orig_clip_y_preds, autocorrect_dict

    def test(self, test_names, epoch=None):
        torch.cuda.synchronize()

        self.args.test_name += '_' + str(epoch)

        test_names = test_names or []

        # Initialize self.meters
        avg_batch_time = AverageMeter()
        avg_data_time = AverageMeter()
        self.meters = defaultdict(AverageMeter)

        self.model.eval()

        end = time.time()

        results = []

        if 'retrieval' in test_names:
            self.cnn_reprs = []
            self.xfmr_reprs = []
            self.fn_ts = []

        with torch.set_grad_enabled(False), tqdm(self.val_loader,
                                                 desc=f'Testing on {", ".join((test_names or []) + ["accuracy and loss"])}',
                                                 disable=self.args.local_rank > 0) as t:
            for batch_idx, (videos, clip_ys, lens, ys, idxs, video_ts) in enumerate(t):
                # Measure data loading time
                avg_data_time.update(time.time() - end)
                end = time.time()

                videos = videos.to(self.device)
                lens = lens.to(self.device)
                clip_ys = clip_ys.to(self.device)

                idxs_repeated = idxs.gather(0, torch.tensor([i for i, l in enumerate(lens) for _ in range(l)])).to(
                    self.device)

                clip_loss, nsp_loss, clip_acc, nsp_acc, clip_preds, nsp_preds, clip_ys, nsp_ys, lens_, self.cnn_reprs_, \
                    xfmr_reprs_, raw_xfmr_reprs, idxs_, video_ts_ = self.model(videos, lens, clip_ys,
                                                                               idxs=idxs_repeated,
                                                                               video_ts=video_ts)

                # loss = clip_loss + self.args.nsp_loss_lambda * nsp_loss
                loss = clip_loss

                if 'retrieval' in test_names:
                    self.cnn_reprs.extend(map(lambda x: x.detach().cpu(), self.cnn_reprs_))
                    self.xfmr_reprs.extend(map(lambda x: x.detach().cpu(), xfmr_reprs_))
                    video_ts_ = video_ts.detach().cpu().split(lens.tolist())
                    self.fn_ts.extend(zip(map(self.val_loader.dataset.get_filename_by_idx, idxs), video_ts_))

                if loss != loss:
                    print('NaN loss, ignoring')
                    continue

                self.meters['clip_loss'].update(clip_loss.item(), sum(clip_ys != -1).item())
                self.meters['nsp_loss'].update(nsp_loss.item(), sum(nsp_ys != -1).item())
                self.meters['clip_acc'].update(clip_acc, sum(clip_ys != -1).item())
                self.meters['nsp_acc'].update(nsp_acc, sum(nsp_ys != -1).item())
                # Measure elapsed time

                if self.args.save_test_results:
                    if 'entailment' in test_names:
                        for idx, ts, preds, gt in zip(idxs_, video_ts_, nsp_preds, nsp_ys):
                            try:
                                idx0, idx1 = [i for i, _ in itertools.groupby(idx.tolist()) if i != -1]
                                ts0, ts1 = [list(y) for x, y in itertools.groupby(ts.tolist(), lambda z: z == -1) if
                                    not x]
                            except ValueError:
                                idx0 = [i for i, _ in itertools.groupby(idx.tolist()) if i != -1][0]
                                idx1 = -1
                                ts0 = [list(y) for x, y in itertools.groupby(ts.tolist(), lambda z: z == -1) if
                                    not x][0]
                                ts1 = -1
                            results.append({
                                'fn0': self.val_loader.dataset.get_filename_by_idx(idx0),
                                'fn1': self.val_loader.dataset.get_filename_by_idx(idx1) if idx1 >= 0 else '',
                                'ts0': ts0,
                                'ts1': ts1,
                                'pred_probs': preds.tolist(),
                                'pred': preds.argmax().item(),
                                'gt': gt.item()
                            })
                    else:
                        clip_ys = clip_ys.chunk(len(ys))
                        # if self.args.local_rank <= 0:
                        #     ipdb.set_trace()
                        video_ts = video_ts.split(lens.tolist())
                        clip_preds = clip_preds.chunk(len(ys))
                        for idx, ts, preds, gts in zip(idxs, video_ts, clip_preds, clip_ys):
                            preds_ = preds.argmax(dim=1)[gts != -1].tolist()
                            pred_probs = preds[gts != -1].tolist()
                            gts = gts[gts != -1].tolist()
                            # nsp_preds_ = nsp_predss.argmax()[nsp_gtss != -1].tolist()
                            # nsp_pred_probs = nsp_predss[nsp_gtss != -1].tolist()
                            # nsp_gtss = nsp_gtss[nsp_gtss != -1].tolist()
                            fn = self.val_loader.dataset.get_filename_by_idx(idx)
                            ts = ts.tolist()
                            results.append({
                                'fn': fn,
                                'ts': ts,
                                'preds': preds_,
                                'pred_probs': pred_probs,
                                'gts': gts,
                                # 'nsp_preds': nsp_preds_,
                                # 'nsp_pred_probs': nsp_pred_probs,
                                # 'nsp_gts': nsp_gtss
                            })

                if 'retrieval' in test_names and batch_idx % self.args.log_freq == 0 and batch_idx:
                    self.cnn_reprs = torch.cat(self.cnn_reprs)
                    self.cnn_reprs = self.cnn_reprs / self.cnn_reprs.norm(p=2, dim=-1).unsqueeze(-1)
                    self.xfmr_reprs = torch.cat(self.xfmr_reprs)
                    self.xfmr_reprs = self.xfmr_reprs / self.xfmr_reprs.norm(p=2, dim=-1).unsqueeze(-1)
                    os.makedirs(os.path.join(self.args.results_dir, self.args.test_name), exist_ok=True)
                    torch.save({
                        'self.cnn_reprs': self.cnn_reprs,
                        'self.xfmr_reprs': self.xfmr_reprs,
                        'self.fn_ts': self.fn_ts
                    },
                        os.path.join(self.args.results_dir, self.args.test_name,
                                     f'retrievals_{self.args.local_rank}_{batch_idx:05}.pt'))
                    del self.cnn_reprs
                    del self.xfmr_reprs
                    del self.fn_ts
                    self.cnn_reprs = []
                    self.xfmr_reprs = []
                    self.fn_ts = []

                avg_batch_time.update(time.time() - end)
                end = time.time()

                t.set_postfix(
                    t_data=avg_data_time.avg,
                    t_batch=avg_batch_time.avg,
                    **{
                        k: v.avg for k, v in self.meters.items()
                    }
                )

        if 'retrieval' in test_names:
            self.cnn_reprs = torch.cat(self.cnn_reprs)
            self.cnn_reprs = self.cnn_reprs / self.cnn_reprs.norm(p=2, dim=-1).unsqueeze(-1)
            self.xfmr_reprs = torch.cat(self.xfmr_reprs)
            self.xfmr_reprs = self.xfmr_reprs / self.xfmr_reprs.norm(p=2, dim=-1).unsqueeze(-1)
            os.makedirs(os.path.join(self.args.results_dir, self.args.test_name), exist_ok=True)
            torch.save({
                'self.cnn_reprs': self.cnn_reprs,
                'self.xfmr_reprs': self.xfmr_reprs,
                'self.fn_ts': self.fn_ts
            },
                os.path.join(self.args.results_dir, self.args.test_name,
                             f'retrievals_{self.args.local_rank}_{batch_idx:05}.pt'))

        gathered_metrics = {}
        for k, v in self.meters.items():
            gathered_metrics[k] = utils.gather_score(v.avg, v.count)

        if self.args.local_rank <= 0:
            print(gathered_metrics)

        if self.args.save_test_results:
            os.makedirs(os.path.join(self.args.results_dir, self.args.test_name), exist_ok=True)
            with open(os.path.join(self.args.results_dir, self.args.test_name,
                                   f'results_{self.args.local_rank}{"_entailment" if "entailment" in test_names else ""}.json'),
                      'w') as f:
                json.dump(results, f)

        return loss

    '''
    CVPR MODEL CODE
    '''

    def autocorrect_cvpr(self, epoch=None):
        torch.cuda.synchronize()

        self.args.test_name += '_' + str(epoch)

        # Initialize self.meters
        avg_batch_time = AverageMeter()
        avg_data_time = AverageMeter()
        self.meters = defaultdict(AverageMeter)

        self.model.eval()

        end = time.time()

        self.cnn_reprs = []
        self.autocorrect_reprs = []
        self.autocorrect_reprs_perstep = []
        self.fn_ts = []

        results = []

        with torch.set_grad_enabled(True), tqdm(self.val_loader,
                                                desc=f'Running autocorrect CVPR',
                                                disable=self.args.local_rank > 0) as t:
            for batch_idx, (videos, clip_ys, lens, ys, idxs, video_ts) in enumerate(t):
                # Measure data loading time
                avg_data_time.update(time.time() - end)
                end = time.time()

                videos = videos.to(self.device)
                lens = lens.to(self.device)
                clip_ys = clip_ys.to(self.device)

                auto_idx, autocorrect_stats, clip_y_preds, orig_clip_y_preds, _ = self.autocorrect_batch_cvpr(clip_ys,
                                                                                                              idxs,
                                                                                                              lens,
                                                                                                              video_ts,
                                                                                                              videos)

                # if self.args.local_rank <= 0:
                #     import ipdb
                #     ipdb.set_trace()

                self.meters['n_iters'].update(auto_idx)
                self.meters['clip_acc'].update(autocorrect_stats['clip_acc'][-1])
                for n in [5, 10, 15, 20, 25]:
                    self.meters[f'success_at_{n}'].update(100 * (auto_idx <= n))
                self.meters[f'reached_100_clip_acc'].update(100 * (autocorrect_stats['clip_acc'][-1] == 100))
                self.meters['delta_preoops_acc'].update(
                    autocorrect_stats['preoops_clip_acc'][-1] - autocorrect_stats['preoops_clip_acc'][0])

                avg_batch_time.update(time.time() - end)
                end = time.time()

                if self.args.save_test_results:
                    clip_ys = clip_ys.split(lens.tolist())
                    video_ts = video_ts.split(lens.tolist())
                    clip_y_preds = clip_y_preds.split(lens.tolist())
                    orig_clip_y_preds = orig_clip_y_preds.split(lens.tolist())
                    for idx, ts, preds, gts, preds_orig in zip(idxs, video_ts, clip_y_preds, clip_ys,
                                                               orig_clip_y_preds):
                        preds_ = preds.argmax(dim=1).tolist()
                        pred_probs = preds.tolist()
                        preds_orig_ = preds_orig.argmax(dim=1).tolist()
                        pred_probs_orig = preds_orig.tolist()
                        gts = gts.tolist()
                        fn = self.val_loader.dataset.get_filename_by_idx(idx)
                        ts = ts.tolist()
                        results.append({
                            'fn': fn,
                            'ts': ts,
                            'preds_final': preds_,
                            'preds_orig': preds_orig_,
                            'pred_probs': pred_probs,
                            'pred_probs_orig': pred_probs_orig,
                            'gts': gts,
                            **{k: v.val for k, v in self.meters.items()}
                        })

                t.set_postfix(
                    t_data=avg_data_time.avg,
                    t_batch=avg_batch_time.avg,
                    **{
                        k: v.avg for k, v in self.meters.items()
                    }
                )

        with torch.no_grad():
            if 'retrieval' in self.args.test_names:
                self.cnn_reprs = torch.cat(self.cnn_reprs)
                self.cnn_reprs = self.cnn_reprs / self.cnn_reprs.norm(p=2, dim=-1).unsqueeze(-1)
                self.autocorrect_reprs = torch.stack(self.autocorrect_reprs)
                self.autocorrect_reprs = self.autocorrect_reprs / self.autocorrect_reprs.norm(p=2, dim=-1).unsqueeze(-1)
                os.makedirs(os.path.join(self.args.results_dir, self.args.test_name), exist_ok=True)
                torch.save({
                    'self.cnn_reprs': self.cnn_reprs,
                    'self.autocorrect_reprs': self.autocorrect_reprs,
                    'autocorrect_cnn': self.args.autocorrect_cnn_output,
                    'self.fn_ts': self.fn_ts
                }, os.path.join(self.args.results_dir, self.args.test_name, f'retrievals_{self.args.local_rank}.pt'))

        gathered_metrics = {}
        for k, v in self.meters.items():
            gathered_metrics[k] = utils.gather_score(v.avg, v.count)

        if self.args.local_rank <= 0:
            print(gathered_metrics)

        if self.args.save_test_results:
            os.makedirs(os.path.join(self.args.results_dir, self.args.test_name), exist_ok=True)
            with open(os.path.join(self.args.results_dir, self.args.test_name,
                                   f'results_{self.args.local_rank}.json'),
                      'w') as f:
                json.dump(results, f)

    def autocorrect_batch_cvpr(self, clip_ys, idxs, lens, video_ts, videos, svo_mode=False):
        clip_loss, clip_acc, self.cnn_reprs_, clip_y_preds = self.model(videos, lens, clip_ys, override_no_svo=True)
        orig_clip_y_preds = clip_y_preds
        with torch.no_grad():
            if not svo_mode and 'retrieval' in self.args.test_names:
                self.cnn_reprs.extend(map(lambda x: x.detach().cpu(), self.cnn_reprs_))
                video_ts_ = video_ts.detach().cpu().split(lens.tolist())
                self.fn_ts.extend(zip(map(self.val_loader.dataset.get_filename_by_idx, idxs), video_ts_))
        # if self.args.local_rank <= 0:
        #     import ipdb
        #     ipdb.set_trace()
        orig_preoops_acc = 100 * (((clip_y_preds.argmax(dim=1)[clip_ys == 0]) == 0).sum().item() / (
                sum(clip_ys == 0).item() or 1))
        autocorrect_stats = defaultdict(list)
        if not svo_mode:
            self.meters['orig_preoops_acc'].update(orig_preoops_acc, sum(clip_ys == 0).item())
        autocorrect_stats['preoops_clip_acc'].append(orig_preoops_acc)
        y_autocorrect = torch.zeros_like(clip_ys) - 1
        y_autocorrect[clip_ys > 0] = 0
        x_autocorrect = self.cnn_reprs_
        x_autocorrect = x_autocorrect.detach().clone().requires_grad_()
        delta_x = torch.zeros_like(x_autocorrect)
        autocorrect_dict = {}
        for auto_idx in range(self.args.max_autocorrect_iters):
            autocorrect_dict['_clip_embs'] = x_autocorrect + delta_x

            clip_loss, clip_acc, _, clip_y_preds, *_ = self.model(None, None, y_autocorrect, **autocorrect_dict,
                                                                  override_no_svo=True)

            autocorrect_loss = clip_loss
            preoops_clip_acc = (100 * (clip_y_preds.argmax(dim=1)[clip_ys == 0] == 0).sum().item() / (
                    (clip_ys == 0).sum().item() or 1))

            autocorrect_stats['clip_loss'].append(clip_loss.item())
            autocorrect_stats['clip_acc'].append(clip_acc)
            autocorrect_stats['preoops_clip_acc'].append(preoops_clip_acc)

            if clip_acc == 100:
                break

            (grad,) = torch.autograd.grad(autocorrect_loss, x_autocorrect)

            delta_x[y_autocorrect == 0] -= self.args.autocorrect_alpha * \
                                           torch.sign(grad)[y_autocorrect == 0]

            delta_x.clamp_(-self.args.autocorrect_eps, self.args.autocorrect_eps)

            # x_autocorrect.grad.data.zero_()
        x_autocorrect = x_autocorrect + delta_x
        with torch.no_grad():
            if not svo_mode and 'retrieval' in self.args.test_names:
                self.autocorrect_reprs.extend(x_autocorrect.detach().cpu())
        return auto_idx, autocorrect_stats, clip_y_preds, orig_clip_y_preds, autocorrect_dict

    def test_cvpr(self, test_names, epoch=None):
        torch.cuda.synchronize()

        self.args.test_name += '_' + str(epoch)

        test_names = test_names or []

        # Initialize self.meters
        avg_batch_time = AverageMeter()
        avg_data_time = AverageMeter()
        self.meters = defaultdict(AverageMeter)

        self.model.eval()

        end = time.time()

        results = []

        if 'retrieval' in test_names:
            self.cnn_reprs = []
            self.fn_ts = []

        with torch.set_grad_enabled(False), tqdm(self.val_loader,
                                                 desc=f'Testing CVPR on {", ".join((test_names or []) + ["accuracy and loss"])}',
                                                 disable=self.args.local_rank > 0) as t:
            for batch_idx, (videos, clip_ys, lens, ys, idxs, video_ts) in enumerate(t):
                # Measure data loading time
                avg_data_time.update(time.time() - end)
                end = time.time()

                videos = videos.to(self.device)
                lens = lens.to(self.device)
                clip_ys = clip_ys.to(self.device)

                clip_loss, clip_acc, self.cnn_reprs_, clip_preds = self.model(videos, lens, clip_ys)

                # loss = clip_loss + self.args.nsp_loss_lambda * nsp_loss
                loss = clip_loss

                if 'retrieval' in test_names:
                    self.cnn_reprs.extend(map(lambda x: x.detach().cpu(), self.cnn_reprs_))
                    video_ts_ = video_ts.detach().cpu().split(lens.tolist())
                    self.fn_ts.extend(zip(map(self.val_loader.dataset.get_filename_by_idx, idxs), video_ts_))

                if loss != loss:
                    print('NaN loss, ignoring')
                    continue

                self.meters['clip_loss'].update(clip_loss.item(), sum(clip_ys != -1).item())
                self.meters['clip_acc'].update(clip_acc, sum(clip_ys != -1).item())
                # Measure elapsed time

                if self.args.save_test_results:
                    clip_ys = clip_ys.split(lens.tolist())
                    # if self.args.local_rank <= 0:
                    #     ipdb.set_trace()
                    video_ts = video_ts.split(lens.tolist())
                    clip_preds = clip_preds.split(lens.tolist())
                    for idx, ts, preds, gts in zip(idxs, video_ts, clip_preds, clip_ys):
                        preds_ = preds.argmax(dim=1).tolist()
                        pred_probs = preds.tolist()
                        gts = gts.tolist()
                        # nsp_preds_ = nsp_predss.argmax()[nsp_gtss != -1].tolist()
                        # nsp_pred_probs = nsp_predss[nsp_gtss != -1].tolist()
                        # nsp_gtss = nsp_gtss[nsp_gtss != -1].tolist()
                        fn = self.val_loader.dataset.get_filename_by_idx(idx)
                        ts = ts.tolist()
                        results.append({
                            'fn': fn,
                            'ts': ts,
                            'preds': preds_,
                            'pred_probs': pred_probs,
                            'gts': gts,
                            # 'nsp_preds': nsp_preds_,
                            # 'nsp_pred_probs': nsp_pred_probs,
                            # 'nsp_gts': nsp_gtss
                        })

                if 'retrieval' in test_names and batch_idx % self.args.log_freq == 0 and batch_idx:
                    self.cnn_reprs = torch.cat(self.cnn_reprs)
                    self.cnn_reprs = self.cnn_reprs / self.cnn_reprs.norm(p=2, dim=-1).unsqueeze(-1)
                    os.makedirs(os.path.join(self.args.results_dir, self.args.test_name), exist_ok=True)
                    torch.save({
                        'self.cnn_reprs': self.cnn_reprs,
                        'self.fn_ts': self.fn_ts
                    },
                        os.path.join(self.args.results_dir, self.args.test_name,
                                     f'retrievals_{self.args.local_rank}_{batch_idx:05}.pt'))
                    del self.cnn_reprs
                    del self.fn_ts
                    self.cnn_reprs = []
                    self.fn_ts = []

                avg_batch_time.update(time.time() - end)
                end = time.time()

                t.set_postfix(
                    t_data=avg_data_time.avg,
                    t_batch=avg_batch_time.avg,
                    **{
                        k: v.avg for k, v in self.meters.items()
                    }
                )

        if 'retrieval' in test_names:
            self.cnn_reprs = torch.cat(self.cnn_reprs)
            self.cnn_reprs = self.cnn_reprs / self.cnn_reprs.norm(p=2, dim=-1).unsqueeze(-1)
            os.makedirs(os.path.join(self.args.results_dir, self.args.test_name), exist_ok=True)
            torch.save({
                'self.cnn_reprs': self.cnn_reprs,
                'self.fn_ts': self.fn_ts
            },
                os.path.join(self.args.results_dir, self.args.test_name,
                             f'retrievals_{self.args.local_rank}_{batch_idx:05}.pt'))

        gathered_metrics = {}
        for k, v in self.meters.items():
            gathered_metrics[k] = utils.gather_score(v.avg, v.count)

        if self.args.local_rank <= 0:
            print(gathered_metrics)

        if self.args.save_test_results:
            os.makedirs(os.path.join(self.args.results_dir, self.args.test_name), exist_ok=True)
            with open(os.path.join(self.args.results_dir, self.args.test_name,
                                   f'results_{self.args.local_rank}{"_entailment" if "entailment" in test_names else ""}.json'),
                      'w') as f:
                json.dump(results, f)

        return loss

    '''
    SVO CODE
    '''

    def test_svo_autocorrect(self, test_names, epoch=None):
        torch.cuda.synchronize()

        self.args.test_name += '_' + str(epoch)

        # Initialize self.meters
        avg_batch_time = AverageMeter()
        avg_data_time = AverageMeter()
        self.meters = defaultdict(AverageMeter)

        self.model.eval()

        results = []

        end = time.time()
        loader = self.val_loader

        with torch.set_grad_enabled(False), tqdm(loader,
                                                 desc=f'Evaluating SVO epoch {epoch}',
                                                 disable=self.args.local_rank > 0) as t:
            for batch_idx, (videos, clip_ys, svo_ys, lens, idxs, video_ts) in enumerate(t):
                # Measure data loading time
                avg_data_time.update(time.time() - end)
                end = time.time()

                videos = videos.to(self.device)
                lens = lens.to(self.device)
                svo_ys = svo_ys.to(self.device)
                clip_ys = clip_ys.to(self.device)

                self.meters['cast_data'].update(time.time() - end)
                end = time.time()

                try:
                    goal_idx = clip_ys.tolist().index(0)
                    svo_ys_all_goal = svo_ys[goal_idx][None].repeat((len(svo_ys), 1, 1))
                    # svo_ys_all_goal[clip_ys == 0] = -1
                except:
                    goal_idx = None
                    svo_ys_all_goal = None

                try:
                    wentwrong_idx = clip_ys.tolist().index(1)
                    svo_ys_all_wentwrong = svo_ys[wentwrong_idx][None].repeat((len(svo_ys), 1, 1))
                    # svo_ys_all_wentwrong[clip_ys == 0] = -1
                except:
                    try:
                        wentwrong_idx = clip_ys.tolist().index(2)
                        svo_ys_all_wentwrong = svo_ys[wentwrong_idx][None].repeat((len(svo_ys), 1, 1))
                        # svo_ys_all_wentwrong[clip_ys == 0] = -1
                    except:
                        wentwrong_idx = None
                        svo_ys_all_wentwrong = None

                with torch.set_grad_enabled(True):
                    if self.args.svo_ac_linearize:
                        clip_loss, nsp_loss, clip_acc, nsp_acc, clip_y_preds, nsp_y_preds, clip_ys_, nsp_ys, lens_, \
                            self.cnn_reprs_, xfmr_reprs_, raw_xfmr_reprs, *_ = self.model(videos, lens, clip_ys,
                                                                                          override_no_svo=True)
                        y_autocorrect = torch.zeros_like(clip_ys) - 1
                        y_autocorrect[clip_ys > 0] = 0
                        y_autocorrect_ = torch.zeros_like(clip_ys_) - 1
                        y_autocorrect_[clip_ys_ > 0] = 0
                        x_autocorrect = torch.cat(
                            self.cnn_reprs_) if self.args.autocorrect_cnn_output else raw_xfmr_reprs
                        x_autocorrect = x_autocorrect.detach().clone().requires_grad_()
                        idx_first_unint = (clip_ys == 0).sum().item()
                        if not idx_first_unint:
                            continue
                        ac_dict = {}
                        ac_dict['_xfmr_tuple'] = [None, lens_, y_autocorrect_, nsp_ys]
                        avg_step = (x_autocorrect[:, idx_first_unint] - x_autocorrect[:, 0]) / idx_first_unint
                        x_autocorrect[:, idx_first_unint:] = (torch.arange(lens_[0] - idx_first_unint).to(
                            avg_step.device)[:, None] * avg_step[None]) + x_autocorrect[:, idx_first_unint]
                        ac_dict['_xfmr_tuple'][0] = x_autocorrect
                        ac_dict['_xfmr_tuple'][2] = clip_ys_
                    elif self.args.load_cvpr_model:
                        *_, ac_dict = self.autocorrect_batch_cvpr(clip_ys, idxs, lens, video_ts, videos, svo_mode=True)
                    else:
                        *_, ac_dict = self.autocorrect_batch(clip_ys, idxs, lens, video_ts, videos, svo_mode=True)

                if svo_ys_all_goal is not None:
                    preac_goal_ys_out, _ = self.model(videos, lens, clip_ys, svo_ys=svo_ys_all_goal,
                                                      pre_svo_reprs=None,
                                                      svo_eval=True)
                    postac_goal_ys_out, _ = self.model(None, lens, clip_ys, svo_ys=svo_ys_all_goal,
                                                       pre_svo_reprs=None,
                                                       svo_eval=True, **ac_dict)
                    n_lbls = (svo_ys_all_goal != -1).sum().item()
                    delta_goal_ys_out = {}
                    for k in preac_goal_ys_out:
                        delta_goal_ys_out[k] = postac_goal_ys_out[k] - preac_goal_ys_out[k]
                    for k, v in delta_goal_ys_out.items():
                        if 'svo_preds' in k:
                            continue
                        try:
                            v = v[-1].item()
                        except:
                            pass
                        self.meters['delta_goal_' + k].update(v, n_lbls)
                    for k, v in preac_goal_ys_out.items():
                        if 'svo_preds' in k:
                            continue
                        try:
                            v = v[-1].item()
                        except:
                            pass
                        self.meters['preac_goal_' + k].update(v, n_lbls)
                    for k, v in postac_goal_ys_out.items():
                        if 'svo_preds' in k:
                            continue
                        try:
                            v = v[-1].item()
                        except:
                            pass
                        self.meters['postac_goal_' + k].update(v, n_lbls)

                if svo_ys_all_wentwrong is not None:
                    preac_wentwrong_ys_out, _ = self.model(videos, lens, clip_ys, svo_ys=svo_ys_all_wentwrong,
                                                           pre_svo_reprs=None,
                                                           svo_eval=True)
                    postac_wentwrong_ys_out, _ = self.model(None, lens, clip_ys, svo_ys=svo_ys_all_wentwrong,
                                                            pre_svo_reprs=None,
                                                            svo_eval=True, **ac_dict)
                    n_lbls = (svo_ys_all_wentwrong != -1).sum().item()
                    delta_wentwrong_ys_out = {}
                    for k in preac_wentwrong_ys_out:
                        delta_wentwrong_ys_out[k] = postac_wentwrong_ys_out[k] - preac_wentwrong_ys_out[k]
                    for k, v in delta_wentwrong_ys_out.items():
                        if 'svo_preds' in k:
                            continue
                        try:
                            v = v[-1].item()
                        except:
                            pass
                        self.meters['delta_wentwrong_' + k].update(v, n_lbls)
                    for k, v in preac_wentwrong_ys_out.items():
                        if 'svo_preds' in k:
                            continue
                        try:
                            v = v[-1].item()
                        except:
                            pass
                        self.meters['preac_wentwrong_' + k].update(v, n_lbls)
                    for k, v in postac_wentwrong_ys_out.items():
                        if 'svo_preds' in k:
                            continue
                        try:
                            v = v[-1].item()
                        except:
                            pass
                        self.meters['postac_wentwrong_' + k].update(v, n_lbls)

                self.meters['fwd_pass'].update(time.time() - end)
                end = time.time()

                if self.args.save_test_results:
                    segments_per_video = [len([_ for _ in (gt_index(ys), len(ys) - gt_index(ys)) if _ > 0]) for ys in
                        clip_ys.split(lens.tolist())]
                    clip_ys = clip_ys.split(lens.tolist())
                    svo_ys = svo_ys.split(lens.tolist(), dim=1)
                    video_ts = video_ts.split(lens.tolist())
                    try:
                        preds = postac_wentwrong_ys_out['svo_preds_top5'].split(2)
                        for idx, clip_y, svo_y, ts, pred in zip(idxs, clip_ys, svo_ys, video_ts, preds):
                            fn = self.val_loader.dataset.get_filename_by_idx(idx)
                            ts = ts.tolist()
                            results.append({
                                'fn': fn,
                                'ts': ts,
                                'pred': pred[-1].tolist()
                            })
                    except:
                        continue

                self.meters['meter_update'].update(time.time() - end)
                end = time.time()

                # Measure elapsed time
                avg_batch_time.update(time.time() - end)
                end = time.time()

                t.set_postfix(
                    t_data=avg_data_time.avg,
                    t_batch=avg_batch_time.avg,
                    **{
                        k: v.avg for k, v in self.meters.items()
                    }
                )

        gathered_metrics = {}
        for k, v in self.meters.items():
            if 'loss' in k or 'acc' in k or 'rank' in k:
                try:
                    gathered_metrics[k] = utils.gather_score(v.avg, v.count).item()
                except:
                    pass
            elif 'mat' in k:
                v_ = torch.from_numpy(v.avg).to(self.device)
                torch.distributed.all_reduce(v_)
                gathered_metrics[k] = v_.cpu().numpy()

        if self.args.save_test_results:
            os.makedirs(os.path.join(self.args.results_dir, self.args.test_name), exist_ok=True)
            with open(os.path.join(self.args.results_dir, seslf.args.test_name,
                                   f'results_{self.args.local_rank}.json'),
                      'w') as f:
                json.dump(results, f)

        if self.args.local_rank <= 0:
            print(gathered_metrics)

    def test_svo(self, test_names, epoch=None):
        torch.cuda.synchronize()

        self.args.test_name += '_' + str(epoch)

        # Initialize self.meters
        avg_batch_time = AverageMeter()
        avg_data_time = AverageMeter()
        self.meters = defaultdict(AverageMeter)

        self.model.eval()

        results = []

        end = time.time()
        loader = self.val_loader

        with torch.set_grad_enabled(False), tqdm(loader,
                                                 desc=f'Evaluating SVO epoch {epoch}',
                                                 disable=self.args.local_rank > 0) as t:
            for batch_idx, (videos, clip_ys, svo_ys, lens, idxs, video_ts) in enumerate(t):
                # Measure data loading time
                avg_data_time.update(time.time() - end)
                end = time.time()

                videos = videos.to(self.device)
                lens = lens.to(self.device)
                svo_ys = svo_ys.to(self.device)
                clip_ys = clip_ys.to(self.device)

                self.meters['cast_data'].update(time.time() - end)
                end = time.time()

                out, _ = self.model(videos, lens, clip_ys, svo_ys=svo_ys, pre_svo_reprs=None, svo_eval=True)

                self.meters['fwd_pass'].update(time.time() - end)
                end = time.time()

                n_lbls = (svo_ys != -1).sum().item()
                for k, v in out.items():
                    if 'svo_preds' in k:
                        continue
                    try:
                        v = v[v == v].mean().item()
                    except:
                        pass
                    self.meters[k].update(v, n_lbls)

                if self.args.save_test_results:
                    segments_per_video = [len([_ for _ in (gt_index(ys), len(ys) - gt_index(ys)) if _ > 0]) for ys in
                        clip_ys.split(lens.tolist())]
                    clip_ys = clip_ys.split(lens.tolist())
                    svo_ys = svo_ys.split(lens.tolist(), dim=1)
                    video_ts = video_ts.split(lens.tolist())
                    preds = out['svo_preds_top5'].split(segments_per_video)
                    for idx, clip_y, svo_y, ts, pred in zip(idxs, clip_ys, svo_ys, video_ts, preds):
                        fn = self.val_loader.dataset.get_filename_by_idx(idx)
                        ts = ts.tolist()
                        results.append({
                            'fn': fn,
                            'ts': ts,
                            'pred': pred.tolist()
                        })

                self.meters['meter_update'].update(time.time() - end)
                end = time.time()

                # Measure elapsed time
                avg_batch_time.update(time.time() - end)
                end = time.time()

                t.set_postfix(
                    t_data=avg_data_time.avg,
                    t_batch=avg_batch_time.avg,
                    **{
                        k: v.avg for k, v in self.meters.items()
                    }
                )

        gathered_metrics = {}
        for k, v in self.meters.items():
            if 'loss' in k or 'acc' in k or 'rank' in k:
                gathered_metrics[k] = utils.gather_score(v.avg, v.count)
            elif 'mat' in k:
                v_ = torch.from_numpy(v.avg).to(self.device)
                torch.distributed.all_reduce(v_)
                gathered_metrics[k] = v_.cpu().numpy()

        if self.args.local_rank <= 0:
            print(gathered_metrics)

        if self.args.save_test_results:
            os.makedirs(os.path.join(self.args.results_dir, self.args.test_name), exist_ok=True)
            with open(os.path.join(self.args.results_dir, self.args.test_name,
                                   f'results_{self.args.local_rank}.json'),
                      'w') as f:
                json.dump(results, f)

        return gathered_metrics['svo_loss']

    def train_svo(self):
        if self.args.cache_pre_svo_reprs:
            all_train_loader = DataLoader(self.train_loader.dataset, batch_size=self.args.batch_size, shuffle=False,
                                          num_workers=self.args.workers, pin_memory=True,
                                          sampler=UniformClipSamplerMulti(self.train_loader.dataset.video_clips,
                                                                          10000),
                                          collate_fn=self.train_loader.dataset.collate_fn)
            all_val_loader = DataLoader(self.val_loader.dataset, batch_size=self.args.batch_size, shuffle=False,
                                        num_workers=self.args.workers, pin_memory=True,
                                        sampler=UniformClipSamplerMulti(self.val_loader.dataset.video_clips,
                                                                        10000),
                                        collate_fn=self.val_loader.dataset.collate_fn)
            with torch.no_grad():
                for i, loader in enumerate((all_train_loader, all_val_loader)):
                    try:
                        loader.dataset.reprs = torch.load(loader.dataset.reprs,
                                                          os.path.join(os.path.dirname(self.args.checkpoint_dir),
                                                                       'reprs', f'pre_svo_reprs_{name}.pt'))
                    except:
                        loader.dataset.reprs = dict()
                        name = "train" if i == 0 else "val"
                        with tqdm(loader, desc=f'Computing cache for SVO ({name})',
                                  disable=self.args.local_rank > 0) as t:
                            for batch_idx, (videos, clip_ys, svo_ys, lens, idxs) in enumerate(t):
                                videos = videos.to(self.device)
                                lens = lens.to(self.device)
                                svo_ys = svo_ys.to(self.device)
                                clip_ys = clip_ys.to(self.device)

                                *_, pre_svo_reprs = self.model(videos, lens, clip_ys, svo_ys=svo_ys)

                                for idx, pre_svo_repr in zip(idxs.cpu().tolist(),
                                                             pre_svo_reprs.detach().cpu().split(lens.tolist())):
                                    loader.dataset.reprs[idx] = pre_svo_repr

                        print('Gathering and saving...')

                        loader.dataset.reprs_cached = True
                        gathered_reprs = utils.all_gather(loader.dataset.reprs)
                        merged_reprs = {k: v for d in gathered_reprs for k, v in d.items()}
                        if self.args.local_rank <= 0:
                            torch.save(merged_reprs, os.path.join(os.path.dirname(self.args.checkpoint_dir),
                                                                  'reprs', f'pre_svo_reprs_{name}.pt'))

                        print('Saved!')

        for epoch in trange(self.epoch, self.args.epochs, desc='Training SVO model'):
            if self.args.local_rank != -1:
                self.train_loader.sampler.set_epoch(epoch)

            self.run_epoch_svo(epoch)

            val_loss = self.run_epoch_svo(epoch, train=False)

            is_best = val_loss < self.best_loss
            self.best_loss = min(val_loss, self.best_loss)

            if self.args.local_rank <= 0:
                print('Saving checkpoint')
                save_checkpoint(self.model, self.optim, epoch, val_loss, self.args.checkpoint_dir, self.global_step,
                                is_best, amp=amp, args=self.args)

    def run_epoch_svo(self, epoch, train=True):
        torch.cuda.synchronize()

        # Initialize self.meters
        avg_batch_time = AverageMeter()
        avg_data_time = AverageMeter()
        self.meters = defaultdict(AverageMeter)

        if train:
            self.model.train()
        else:
            self.model.eval()

        def bn_to_eval(m):
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                m.training = False
            for module in m.children():
                bn_to_eval(module)
            return m

        bn_to_eval(self.model)

        end = time.time()
        loader = self.train_loader if train else self.val_loader

        with torch.set_grad_enabled(train), tqdm(loader,
                                                 desc=f'Training SVO epoch {epoch}' if train else f'Validating SVO {f"epoch {epoch}" if epoch else ""}',
                                                 disable=self.args.local_rank > 0) as t:
            for batch_idx, (videos, clip_ys, svo_ys, lens, idxs) in enumerate(t):
                # Measure data loading time
                avg_data_time.update(time.time() - end)
                end = time.time()

                videos = videos.to(self.device)
                lens = lens.to(self.device)
                svo_ys = svo_ys.to(self.device)
                clip_ys = clip_ys.to(self.device)

                self.meters['cast_data'].update(time.time() - end)
                end = time.time()

                svo_ys[~torch.full((len(svo_ys),), self.args.svo_pred_prob).bernoulli().bool()] = -1

                pre_svo_reprs = None
                if self.args.cache_pre_svo_reprs:
                    pre_svo_reprs = videos
                    videos = None

                loss, svo_acc, svo_acc_allthree, _ = self.model(videos, lens, clip_ys, svo_ys=svo_ys,
                                                                pre_svo_reprs=pre_svo_reprs)

                self.meters['fwd_pass'].update(time.time() - end)
                end = time.time()

                if loss != loss:
                    print('NaN loss, ignoring')
                    # if self.args.local_rank <= 0:
                    #     import ipdb
                    #     ipdb.set_trace()
                    continue

                n_lbls = (svo_ys != -1).sum().item()
                self.meters['svo_loss'].update(loss.item(), n_lbls)
                self.meters['svo_acc'].update(svo_acc, n_lbls)
                self.meters['svo_acc_allthree'].update(svo_acc_allthree, n_lbls)

                self.meters['meter_update'].update(time.time() - end)
                end = time.time()

                if train:
                    loss = loss / self.args.grad_accumulate_steps

                    if self.args.fp16:
                        with amp.scale_loss(loss, self.optim) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss.backward()

                    self.meters['backward'].update(time.time() - end)
                    end = time.time()

                    if batch_idx and batch_idx % self.args.grad_accumulate_steps == 0:
                        self.optim.step()
                        self.optim.zero_grad()

                    self.meters['optim_step'].update(time.time() - end)
                    end = time.time()

                # Measure elapsed time
                avg_batch_time.update(time.time() - end)
                end = time.time()

                t.set_postfix(
                    t_data=avg_data_time.avg,
                    t_batch=avg_batch_time.avg,
                    **{
                        k: v.avg for k, v in self.meters.items()
                    }
                )

                if train:
                    if self.global_step % self.args.log_freq == 0 and self.writer and not self.args.debug and batch_idx:
                        self.writer.add_scalars('train/loss',
                                                {
                                                    k: v.avg for k, v in self.meters.items() if 'loss' in k
                                                },
                                                self.global_step * self.args.batch_size * self.args.step_n_gpus)
                        self.writer.add_scalars('train/acc',
                                                {
                                                    k: v.avg for k, v in self.meters.items() if 'acc' in k
                                                },
                                                self.global_step * self.args.batch_size * self.args.step_n_gpus)

                    self.global_step += 1

        if train:
            return self.meters['loss'].avg
        if not train:
            gathered_metrics = {}
            for k, v in self.meters.items():
                if 'loss' in k or 'acc' in k:
                    gathered_metrics[k] = utils.gather_score(v.avg, v.count)
                elif 'mat' in k:
                    v_ = torch.from_numpy(v.avg).to(self.device)
                    torch.distributed.all_reduce(v_)
                    gathered_metrics[k] = v_.cpu().numpy()

            if self.args.local_rank <= 0:
                print(gathered_metrics)

            if epoch is not None and self.writer is not None:
                self.writer.add_scalars('val/loss', {k: v for k, v in gathered_metrics.items() if 'loss' in k}, epoch)
                self.writer.add_scalars('val/acc', {k: v for k, v in gathered_metrics.items() if 'acc' in k}, epoch)

            return -gathered_metrics['svo_acc']
