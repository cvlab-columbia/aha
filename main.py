import argparse
import json
import os
import random
import socket
import warnings
from datetime import datetime

from nltk import WordNetLemmatizer
from torch import nn
from torch.optim import AdamW
from torchvision.models.video import r3d_18
from tqdm import tqdm

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets.samplers import DistributedSampler

from datasets import KineticsAndOops, oops_idx_to_label, train_transform, val_transform
from models import AhaModel
from samplers import RandomClipSamplerMulti, UniformClipSamplerMulti
from trainer import Trainer
from utils import load_checkpoint, wordfreq_txt_to_dict
from spellchecker import SpellChecker

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser = argparse.ArgumentParser(description='',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=32, type=int, metavar='N',
                        help='batchsize')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers')
    parser.add_argument('--log_freq', '-f', default=50, type=int, help='print frequency')
    parser.add_argument('--debug', action='store_true', help="Debug (no writing to disk at all)")
    parser.add_argument('-c', '--checkpoint_dir',
                        type=str,
                        metavar='PATH',
                        help='path to save and load checkpoint',
                        default='PATH_TO_YOUR_DIR/checkpoints/autocorrect')
    parser.add_argument('-l', '--log_dir',
                        type=str,
                        metavar='PATH',
                        help='path to save and load tensorboard writers',
                        default='PATH_TO_YOUR_DIR/logs/autocorrect')
    parser.add_argument('-r', '--results_dir',
                        type=str,
                        metavar='PATH',
                        help='path to save test results',
                        default='PATH_TO_YOUR_DIR/results/autocorrect')
    parser.add_argument('-cfg', '--config_path',
                        type=str,
                        metavar='PATH',
                        help='path to save test results',
                        default='PATH_TO_YOUR_DIR/code/autocorrect/config/config.json')
    parser.add_argument('--name')
    parser.add_argument('--resume_name', help='Experiment name from which to resume')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--validate', action='store_true', help='validate model on val set')
    parser.add_argument('--test', action='store_true', help='test model')
    parser.add_argument('--test_names', nargs='*', help='test names')
    parser.add_argument('--resume_latest', action='store_true')
    parser.add_argument('--fused_adam', action='store_true')
    parser.add_argument('--from_pretrained', action='store_true')
    parser.add_argument('--freeze_cnn', action='store_true')
    parser.add_argument("--adam_epsilon", default=1e-4, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument("--learning_rate",
                        default=3e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument('--opt_level', default="O1")
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument(
        '--oops_root', default='PATH_TO_OOPS_DATA')
    parser.add_argument(
        '--kinetics_root', default='PATH_TO_KINETICS_DATA')
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--image_dim', type=int, default=112)
    parser.add_argument('--train_data_fraction', type=float, default=1.0)
    parser.add_argument('--val_data_fraction', type=float, default=1.0)
    parser.add_argument('--use_oops_gt', action='store_true')
    parser.add_argument('--use_pseudo_oops_gt', action='store_true')
    parser.add_argument('--fixed_position_embeddings', action='store_true')
    parser.add_argument('--weighted_clip_loss', action='store_true')
    parser.add_argument('--undersample_clip_loss', action='store_true')
    parser.add_argument('--clip_sample_prob_scale', default=1.0, type=float,
                        help='only applies with --undersample_clip_loss')
    parser.add_argument('--use_oops_only', action='store_true', help="If true, don't use Kinetics data to train")
    parser.add_argument('--use_kinetics_only', action='store_true', help="If true, don't use Oops data to train")
    parser.add_argument('--nsp_loss_lambda', type=float, default=0.5)
    parser.add_argument('--grad_accumulate_steps', type=int, default=1)
    parser.add_argument('--load_cvpr_model', action='store_true')
    parser.add_argument('--resume_cvpr_model', action='store_true')
    parser.add_argument('--cvpr_model_path',
                        default='PATH_TO_CVPR_CHECKPOINT')
    '''
    Trajectory args
    '''
    # How many clips per trajectory. Trajectory duration in sec = (trajectory_size*frames_per_clip + (trajectory_size - 1)*num_clips_between_clips*step_between_clips)/frame_rate
    parser.add_argument('--max_trajectory_size', type=int, default=10)
    parser.add_argument('--min_trajectory_size', type=int, default=10)
    parser.add_argument('--load_entire_video', action='store_true',
                        help='Load entire video as only trajectory.')
    # How many trajectories per video in training/validation
    parser.add_argument('--max_trajectories_per_video', type=int, default=10)
    # Video sampling fps
    parser.add_argument('--frame_rate', type=int, default=16)
    # Clip duration in frames. Duration in sec = frames_per_clip / frame_rate
    parser.add_argument('--frames_per_clip', type=int, default=16)
    # Spacing between start times of adjacent clips in the dataset, in frames. Spacing in sec = step_between_clips / frame_rate
    parser.add_argument('--step_between_clips', type=int, default=4)
    # Spacing between start times of adjacent clips in a trajectory, in clips. Spacing in sec = num_clips_between_clips * step_between_clips / frame_rate
    # Spacing between end of previous clip and start of next one in sec = (num_clips_between_clips * step_between_clips - frames_per_clip) / frame_rate
    parser.add_argument('--num_clips_between_clips', type=int, default=4)

    # Transformer args
    parser.add_argument('--p_no_sep_seq', type=float, default=0.5,
                        help='Probability of not separating video input to the transformer')
    parser.add_argument('--min_xfmr_seq_len', type=int, default=2,
                        help='When separating input into two sequences, minimum sequence size')
    parser.add_argument('--p_swap_seq_order', type=float, default=1 / 3,
                        help='If separating video input, and if deciding that y=0, probability of swapping'
                             ' bisected video instead of using sequence from other video')
    parser.add_argument('--p_nsp_false', type=float, default=0.5,
                        help='probability of NSP label being 0, conditioned on decision to do NSP')
    parser.add_argument('--nsp_split_at_oops', action='store_true',
                        help='If splitting videos, do so at the oops moment')

    # Test args
    parser.add_argument('--save_test_results', action='store_true')
    parser.add_argument('--test_name', default='test')
    # Autocorrect args
    parser.add_argument('--autocorrect_cnn_output', action='store_true',
                        help='If false (default), autocorrect transformer output instead')
    parser.add_argument('--max_autocorrect_iters', default=50, type=int,
                        help='Number of adversarial iterations in autocorrect')
    parser.add_argument('--autocorrect_alpha', type=float, default=0.03)
    parser.add_argument('--autocorrect_eps', type=float, default=1)

    # Decoder args
    parser.add_argument('--svo', action='store_true')
    parser.add_argument('--svo_pretrained_embs', action='store_true')
    parser.add_argument('--svo_freeze_embs', action='store_true')
    parser.add_argument('--svo_ac_linearize', action='store_true')
    parser.add_argument('--svo_compute_rank', action='store_true')
    parser.add_argument('--svo_cnn_only', action='store_true')
    parser.add_argument('--svo_test_autocorrect', action='store_true')
    parser.add_argument('--cache_pre_svo_reprs', action='store_true')
    parser.add_argument('--svo_dim', default=768, type=int)
    parser.add_argument('--svo_path', default='PATH_TO_OOPS_SVO_DATA')
    parser.add_argument('--svo_results_path')
    parser.add_argument('--svo_wordfreq_file', default='en_full.txt')
    parser.add_argument('--svo_pred_prob', type=float, default=0.5)

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()

    seed = 615
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    numpy.random.seed(seed)
    random.seed(seed)

    args.use_oops_gt = True

    args.test_names = args.test_names or []

    if args.svo_test_autocorrect and args.test_name == 'test':
        args.test_name = 'autocorrect'

    if 'autocorrect' in args.test_names:
        args.autocorrect = True
        args.test = False
        if args.test_name == 'test':
            args.test_name = 'autocorrect'
    else:
        args.autocorrect = False

    if args.test:
        if 'entailment' in args.test_names:
            args.p_no_sep_seq = 0
        else:
            args.p_no_sep_seq = 1

    if args.svo:
        args.p_no_sep_seq = 1
        args.use_oops_only = True
        args.use_oops_gt = True
        if args.load_cvpr_model:
            args.svo_cnn_only = True
        if args.svo_test_autocorrect:
            args.p_no_sep_seq = 0
            args.p_nsp_false = 0
            args.nsp_split_at_oops = True
            args.load_entire_video = True
            args.use_oops_only = True
            args.use_oops_gt = True
            args.batch_size = 1
            args.svo_compute_rank = True

    if args.autocorrect:
        args.p_no_sep_seq = 0
        args.p_nsp_false = 0
        args.nsp_split_at_oops = True
        args.load_entire_video = True
        args.use_oops_only = True
        args.use_oops_gt = True
        args.batch_size = 1

        if args.load_cvpr_model:
            args.autocorrect_cnn_output = True

    assert not (args.undersample_clip_loss and args.weighted_clip_loss)
    # assert not (args.test and args.p_no_sep_seq != 1)

    if args.load_entire_video:
        args.min_trajectory_size = 0
        args.max_trajectory_size = 50

    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    if not args.resume:
        assert args.name is not None and len(args.name) > 0
        args.name = args.name + '_' + current_time + '_' + socket.gethostname()
        if args.local_rank <= 0:
            print('Name: ' + args.name)
    else:
        assert args.resume_name is not None and len(args.resume_name) > 0
        args.name = args.resume_name

    args.checkpoint_dir = os.path.join(args.checkpoint_dir, args.name)
    args.log_dir = os.path.join(args.log_dir, args.name)
    args.results_dir = os.path.join(args.results_dir, args.name)

    if args.debug and args.local_rank <= 0:
        print('Debugging!')

    import torchvision

    torchvision.set_video_backend('video_reader')

    try:
        if not args.debug:
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            os.makedirs(args.log_dir, exist_ok=True)
            os.makedirs(args.results_dir, exist_ok=True)
    except Exception as e:
        print('Error creating directories!')
        print(e)

    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()
        args.step_n_gpus = n_gpu
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        torch.distributed.init_process_group(
            backend='nccl', init_method='env://')
        args.step_n_gpus = torch.distributed.get_world_size()

    args_dict = {
        k: args.__getattribute__(k) for k in
        {'num_clips_between_clips', 'frames_per_clip', 'step_between_clips', 'frame_rate', 'use_oops_gt', 'local_rank',
            'use_oops_only', 'use_pseudo_oops_gt', 'use_kinetics_only', 'svo', 'svo_path', 'cache_pre_svo_reprs'}
    }

    if args.svo:
        try:
            with open(os.path.join(args.svo_path, 'vocab.json')) as f:
                vocab = json.load(f)
            with open(os.path.join(args.svo_path, 'corrections.json')) as f:
                corrections = json.load(f)
        except FileNotFoundError:
            svo_data = []
            for fn in ['train', 'val']:
                with open(os.path.join(args.svo_path, fn + '.json')) as f:
                    svo_data.append(json.load(f))
            svo_data = {k: v for d in svo_data for k, v in d.items()}
            vocab = set([word for wordset in [set([word for words in ann['kgoalsvos'] for word in words]) | set(
                [word for words in ann['kwentwrongsvos'] for word in words]) for v in svo_data.values() for ann in v]
                            for
                            word in wordset])
            spell = SpellChecker()
            spell.word_frequency._dictionary.update(
                wordfreq_txt_to_dict(os.path.join(args.svo_path, args.svo_wordfreq_file)))
            spell.word_frequency._update_dictionary()
            corrections = {}
            for w in tqdm(spell.unknown(vocab)):
                if not w or w.startswith('get '):
                    continue
                pre = ''
                w_ = w
                if w.startswith('!'):
                    w_ = w[1:]
                    if all(spell.known(w_)):
                        continue
                    pre = '!'
                if len(w) >= 10:
                    spell.distance = 1
                corr = pre + spell.correction(w_)
                spell.distance = 2
                if w != corr:
                    corrections[w] = corr
                    vocab.remove(w)
                    vocab.add(corr)
            lemmatizer = WordNetLemmatizer()
            for w in tqdm(list(vocab)):
                pre = ''
                w_ = w
                if w.startswith('!'):
                    w_ = w[1:]
                    pre = '!'
                corr = pre + lemmatizer.lemmatize(w_)
                if w != corr:
                    corrections[w] = corr
                    vocab.remove(w)
                    vocab.add(corr)
        args.svo_vocab_size = len(vocab)
        args.svo_vocab = vocab
        args.svo_corrs = corrections


    def get_dataset(mode):
        return KineticsAndOops(args.kinetics_root, args.oops_root, args.min_trajectory_size, args.max_trajectory_size,
                               oops_label_fn=oops_idx_to_label,
                               return_batch_details=args.test or args.autocorrect,
                               entire_video_as_clip=args.load_entire_video,
                               kinetics_label_fn=lambda *_: torch.tensor(0),
                               transform=train_transform(args.image_dim) if mode == 'train' else val_transform(
                                   args.image_dim),
                               # transform=val_transform(args.image_dim),
                               train=mode == 'train',
                               max_clips_per_video=args.max_trajectories_per_video,
                               data_fraction=args.train_data_fraction if mode == 'train' else args.val_data_fraction,
                               **args_dict)


    # if args.svo and args.test:
    #     train_loader = None
    #     val_data = SVOTestDataset(args)
    #
    #     val_sampler = None
    #     if args.local_rank != -1:  # Distributed training
    #         val_sampler = DistributedSampler(val_data)
    #
    #     val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False,
    #                             num_workers=args.workers, pin_memory=True, sampler=val_sampler,
    #                             collate_fn=val_data.collate_fn)

    train_data = get_dataset('train')
    val_data = get_dataset('val')

    train_sampler = RandomClipSamplerMulti(train_data.video_clips, args.max_trajectories_per_video)
    val_sampler = UniformClipSamplerMulti(val_data.video_clips,
                                          args.max_trajectories_per_video)

    if args.local_rank != -1:  # Distributed training
        train_sampler = DistributedSampler(train_sampler)
        val_sampler = DistributedSampler(val_sampler)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=True, sampler=train_sampler,
                              collate_fn=train_data.collate_fn)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True, sampler=val_sampler,
                            collate_fn=val_data.collate_fn)

    if args.fp16:
        try:
            import apex

            convert_sync_batchnorm = apex.parallel.convert_syncbn_model
            print('Using apex sync batchnorm')
        except:
            print('Using pytorch sync batchnorm')
            convert_sync_batchnorm = torch.nn.SyncBatchNorm.convert_sync_batchnorm
    else:
        print('Using pytorch sync batchnorm')
        convert_sync_batchnorm = torch.nn.SyncBatchNorm.convert_sync_batchnorm

    # tok = BertTokenizer.from_pretrained('bert-base-uncased')
    model = AhaModel(args)
    if args.local_rank > -1:
        model = convert_sync_batchnorm(model)
    model.to(device)

    if args.load_cvpr_model:
        cnn = r3d_18(pretrained=args.from_pretrained)
        cnn.fc = nn.Linear(512, 3)
        model.video_transformer.embeddings.clip_embeddings = cnn.to(device)

    if args.freeze_cnn:
        for n, p in model.named_parameters():
            p.requires_grad = 'clip_embeddings' not in n

    if args.svo:
        for n, p in model.named_parameters():
            p.requires_grad = n.startswith('svo')
        if args.svo_freeze_embs:
            for p in model.svo_decoder_embs.parameters():
                p.requires_grad = False

    if (args.svo and args.svo_cnn_only) or args.load_cvpr_model:
        model.cnn = model.video_transformer.embeddings.clip_embeddings
        del model.video_transformer
        del model.clip_prediction
        del model.next_seq_prediction

    if args.fp16:
        try:
            from apex.optimizers import FusedAdam
            from apex import amp, optimizers
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        params = model.parameters()
        if args.svo:
            params = [p for p in params if p.requires_grad]
        if args.fused_adam:
            args.opt_level = "O1"
            args.loss_scale = None
            args.keep_batchnorm_fp32 = None
            optim = FusedAdam(params, lr=args.learning_rate, eps=args.adam_epsilon)
        else:
            args.keep_batchnorm_fp32 = None
            optim = AdamW(params, lr=args.learning_rate, eps=args.adam_epsilon)

        if args.loss_scale == 0:
            args.loss_scale = None

        if args.opt_level == "O1":
            args.keep_batchnorm_fp32 = None
            args.loss_scale = "dynamic"

        model, optim = amp.initialize(model, optim, opt_level=args.opt_level, loss_scale=args.loss_scale,
                                      keep_batchnorm_fp32=args.keep_batchnorm_fp32)
    else:
        amp = None
        optim = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)

    if not args.autocorrect:
        if args.local_rank != -1:
            try:
                # raise Exception('apex parallel doesnt mesh well with pytorch 1.2')
                from apex.parallel import DistributedDataParallel as DDP

                ddp_kwargs = {'delay_allreduce': args.autocorrect}
                # ddp_kwargs = {}
            except:
                from torch.nn.parallel import DistributedDataParallel as DDP

                ddp_kwargs = {'device_ids': [args.local_rank], 'output_device': args.local_rank}

                print('Using PyTorch DDP')
            model = DDP(model, **ddp_kwargs)
        elif n_gpu > 1:
            model = torch.nn.DataParallel(model)

    epoch = global_step = 0
    best_loss = float("inf")

    if args.load_cvpr_model:
        args.nsp_loss_lambda = 0
        model_ = model
        if args.local_rank != -1 and not args.autocorrect:
            model_ = model_.module
        if args.resume_cvpr_model:
            model_to_load = torch.load(os.path.join(args.cvpr_model_path, 'model_best.pth'), map_location=device)
            try:
                model_.cnn.load_state_dict(model_to_load['state_dict'])
            except RuntimeError:  # dataparallel vs regular
                d = {}
                for k, v in model_to_load['state_dict'].items():
                    d[k.replace('module.', '')] = v
                model_.cnn.load_state_dict(d)
        model_.cvpr_oops_decoder = model_.cnn.fc
        model_.cnn.fc = nn.Sequential()

    if args.resume:
        try:
            epoch, best_loss, global_step = load_checkpoint(model, optim, args.checkpoint_dir, device, amp=amp,
                                                            filename='checkpoint.pth' if args.resume_latest else 'model_best.pth')
            if args.test or args.autocorrect or args.validate:
                epoch -= 1
        except FileNotFoundError:
            if args.local_rank <= 0:
                print('Checkpoint not found!')
        if args.local_rank <= 0:
            print('Resumed from checkpoint, now at epoch', epoch)

    writer = SummaryWriter(
        log_dir=args.log_dir) if args.local_rank <= 0 and not args.validate and not args.debug else None

    trainer = Trainer(model=model, optim=optim, train_loader=train_loader, val_loader=val_loader, args=args,
                      epoch=epoch, writer=writer, global_step=global_step, best_loss=best_loss, device=device,
                      tok=None)
    if args.svo:
        if args.test:
            trainer.val_loader.dataset.oops.svo_eval = True
            if args.svo_test_autocorrect:
                trainer.test_svo_autocorrect(test_names=args.test_names, epoch=epoch)
            else:
                trainer.test_svo(test_names=args.test_names, epoch=epoch)
        elif args.validate:
            trainer.run_epoch_svo(epoch=epoch, train=False)
        else:
            trainer.train_svo()
    elif args.test:
        if args.local_rank <= 0:
            print(f'Testing on {", ".join((args.test_names or []) + ["accuracy and loss"])}')
        if args.load_cvpr_model:
            trainer.test_cvpr(test_names=args.test_names, epoch=epoch)
        else:
            trainer.test(test_names=args.test_names, epoch=epoch)
    elif args.autocorrect:
        if args.load_cvpr_model:
            trainer.autocorrect_cvpr(epoch=epoch)
        else:
            trainer.autocorrect(epoch=epoch)
    elif args.validate:
        trainer.run_epoch(train=False, epoch=None)
    else:
        trainer.train()
