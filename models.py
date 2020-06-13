# class DeOopsifier(nn.Module):
#     def __init__(self, args):
#         super().__init__()
#         if args.from_pretrained:
#             print('Starting from pretrained model')
#         self.model = r2plus1d_18(pretrained=args.from_pretrained)
#         self.model.fc = nn.Linear(in_features=512, out_features=args.hidden_dim)
#
#     def forward(self, data):
#         # data: [N x K x C x T x H x W]
#         # out: [N x K x HIDDEN_DIM]
#         n, k, *_ = data.shape
#         # out = [self.model(data[:, i]) for i in range(data.shape[1])]
#         out = self.model(data.view(n*k, *_)).view(n, k, -1)
#         return out
#
import itertools
import math
import random

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from torchvision.models.video import r2plus1d_18
from transformers import BertConfig, BertPreTrainedModel, BertTokenizer
from transformers.modeling_bert import BertEmbeddings, BertLayerNorm, BertEncoder, ACT2FN, BertModel


def gt_index(l):
    return len([_ for _ in l if _ == 0])


# from https://nlp.seas.harvard.edu/2018/04/03/attention.html
class PositionEmbeddings(nn.Module):
    def __init__(self, d_model, dropout, max_len=64):
        super(PositionEmbeddings, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


class VideoTransformerEmbedder(BertEmbeddings):
    def __init__(self, cfg, args, tok: BertTokenizer):
        super().__init__(cfg)
        self.clip_embeddings = r2plus1d_18(pretrained=args.from_pretrained)
        self.clip_embeddings.fc = nn.Linear(in_features=512, out_features=cfg.hidden_size)
        self.LayerNorm = BertLayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.dropout = nn.Dropout(cfg.hidden_dropout_prob)
        self.tok = tok
        self.args = args
        if args.fixed_position_embeddings:
            self.position_embeddings = PositionEmbeddings(cfg.hidden_size, cfg.hidden_dropout_prob)

    def forward(self, x, lens, clip_ys, idxs, video_ts, _clip_embs=None):
        # x: [(N * sum K_i) x C x T x  H x W]
        if _clip_embs is not None:
            x = _clip_embs
            clip_embs = _clip_embs
            d = 1
        else:
            lens = lens.tolist()
            x = self.clip_embeddings(x)
            x = x.split(lens)
            clip_embs = x
            d = 0

        try:
            idxs = idxs.split(lens)
            video_ts = video_ts.split(lens)
        except AttributeError:
            pass

        clip_ys, lens, nsp_ys, x, idxs, video_ts = self.process_embeddings(x, clip_ys.split(lens), idxs, video_ts,
                                                                           d_clip_split=d)

        if self.args.fixed_position_embeddings:
            x = self.position_embeddings(x)
        else:
            x = x + self.position_embeddings(torch.arange(x.shape[1], dtype=torch.long, device=x.device).clamp(
                max=self.position_embeddings.weight.shape[0] - 1)).unsqueeze(0).expand_as(x)
        # x: [N x K x HIDDEN_DIM]
        x = self.LayerNorm(x)
        x = self.dropout(x)
        return x, lens, clip_ys, nsp_ys, clip_embs, idxs, video_ts

    def process_embeddings(self, x, clip_ys, idxs, video_ts, d_clip_split=0):
        # process data, adding special tokens
        device = x[0].device
        cls_sep_pad_emb = self.word_embeddings(torch.LongTensor([[0, 1, 2]]).to(device)) + \
                          self.token_type_embeddings(torch.LongTensor([[2, 2, 2]]).to(device))
        cls_emb = cls_sep_pad_emb[:, 0]
        sep_emb = cls_sep_pad_emb[:, 1]
        pad_emb = cls_sep_pad_emb[:, 2]
        d = cls_emb.shape[1]
        x_separated = []
        do_idxs = idxs is not None
        do_video_ts = video_ts is not None
        clip_ys_separated = []
        idxs_separated = [] if do_idxs else [None for _ in x]
        video_ts_separated = [] if do_video_ts else [None for _ in x]
        nsp_ys = []
        for i, x_ in enumerate(x):
            if random.random() > self.args.p_no_sep_seq:
                if self.args.nsp_split_at_oops:
                    split_i = (clip_ys[i] + d_clip_split == 0).sum()
                else:
                    try:
                        split_i = random.randint(self.args.min_xfmr_seq_len, len(x_) - self.args.min_xfmr_seq_len)
                    except:
                        split_i = random.randint(len(x_) - self.args.min_xfmr_seq_len, self.args.min_xfmr_seq_len)
                x_l, x_r = x_[:split_i], x_[split_i:]
                clip_y_l, clip_y_r = clip_ys[i][:split_i], clip_ys[i][split_i:]
                if do_idxs: idx_l, idx_r = idxs[i][:split_i], idxs[i][split_i:]
                if do_video_ts: video_t_l, video_t_r = video_ts[i][:split_i], video_ts[i][split_i:]
                if random.random() < self.args.p_nsp_false:
                    nsp_ys.append(0)  # R does not follow L
                    if random.random() < self.args.p_swap_seq_order:  # Don't mine negative from other video in batch, just swap the bisected segments
                        x_l, x_r = x_r, x_l
                        clip_y_l, clip_y_r = clip_y_r, clip_y_l
                        if do_idxs: idx_l, idx_r = idx_r, idx_l
                        if do_video_ts: video_t_l, video_t_r = video_t_r, video_t_l
                    else:
                        try:
                            i_other = random.choice([_ for _ in range(len(x)) if _ != i])
                            x_other = x[i_other]
                            clip_y_other = clip_ys[i_other]
                            if do_idxs: idx_other = idxs[i_other]
                            if do_video_ts: video_t_other = video_ts[i_other]
                            try:
                                split_i_other = random.randint(self.args.min_xfmr_seq_len,
                                                               len(x_other) - self.args.min_xfmr_seq_len)
                            except:
                                split_i_other = random.randint(len(x_other) - self.args.min_xfmr_seq_len,
                                                               self.args.min_xfmr_seq_len)
                            if random.random() < 0.5:  # Swap L
                                max_len = self.args.max_trajectory_size - len(x_r)
                                x_l = x_other[:split_i_other][:max_len]
                                clip_y_l = clip_y_other[:split_i_other][:max_len]
                                if do_idxs: idx_l = idx_other[:split_i_other][:max_len]
                                if do_video_ts: video_t_l = video_t_other[:split_i_other][:max_len]
                            else:  # Swap R
                                max_len = self.args.max_trajectory_size - len(x_l)
                                x_r = x_other[split_i_other:][:max_len]
                                clip_y_r = clip_y_other[split_i_other:][:max_len]
                                if do_idxs: idx_r = idx_other[split_i_other:][:max_len]
                                if do_video_ts: video_t_r = video_t_other[split_i_other:][:max_len]
                        except IndexError:
                            x_l, x_r = x_r, x_l
                            clip_y_l, clip_y_r = clip_y_r, clip_y_l
                            if do_idxs: idx_l, idx_r = idx_r, idx_l
                            if do_video_ts: video_t_l, video_t_r = video_t_r, video_t_l
                else:
                    nsp_ys.append(1)  # R does follow L
                x_l = x_l + self.token_type_embeddings(torch.zeros(len(x_l), dtype=int, device=device))
                x_r = x_r + self.token_type_embeddings(torch.ones(len(x_r), dtype=int, device=device))
                x_separated.append([cls_emb, x_l, sep_emb, x_r, sep_emb])
                clip_ys_separated.append([-1] + clip_y_l.tolist() + [-1] + clip_y_r.tolist() + [-1])
                if do_idxs: idxs_separated.append([-1] + idx_l.tolist() + [-1] + idx_r.tolist() + [-1])
                if do_video_ts: video_ts_separated.append([-1] + video_t_l.tolist() + [-1] + video_t_r.tolist() + [-1])
            else:
                x_ = x_ + self.token_type_embeddings(torch.zeros(len(x_), dtype=int, device=device))
                x_separated.append([cls_emb, x_, sep_emb])
                clip_ys_separated.append([-1] + clip_ys[i].tolist() + [-1])
                if do_idxs: idxs_separated.append([-1] + idxs[i].tolist() + [-1])
                if do_video_ts: video_ts_separated.append([-1] + video_ts[i].tolist() + [-1])
                nsp_ys.append(-1)
        lens = torch.LongTensor([sum(len(x_) for x_ in x_list) for x_list in x_separated])  # [N]
        for l, x_list, clip_y_list, idx_list, video_t_list in zip(lens, x_separated, clip_ys_separated, idxs_separated,
                                                                  video_ts_separated):
            x_list.extend([pad_emb for _ in range(max(lens) - l)])
            clip_y_list.extend([-1 for _ in range(max(lens) - l)])
            if do_idxs: idx_list.extend([-1 for _ in range(max(lens) - l)])
            if do_video_ts: video_t_list.extend([-1 for _ in range(max(lens) - l)])
        x = torch.stack([torch.cat(_) for _ in x_separated])  # [N x max K_i x HIDDEN_DIM]
        clip_ys = torch.stack([torch.LongTensor(_) for _ in clip_ys_separated]).to(device)  # [N x max K_i]
        nsp_ys = torch.LongTensor(nsp_ys).to(device)  # [N]
        if do_idxs: idxs = torch.stack([torch.LongTensor(_) for _ in idxs_separated]).to(device)
        if do_video_ts: video_ts = torch.stack([torch.tensor(_) for _ in video_ts_separated]).to(device)
        return clip_ys, lens, nsp_ys, x, idxs, video_ts


class VideoTransformerHead(nn.Module):
    def __init__(self, d_in, d_out, hidden_act='gelu', pool=None):
        super().__init__()
        self.pool = pool
        if isinstance(hidden_act, str):
            self.transform_act_fn = ACT2FN[hidden_act]
        else:
            self.transform_act_fn = hidden_act
        self.linear = nn.Linear(d_in, d_out)
        self.pool_fns = {
            'avg': self.pool_avg,
            'first': self.pool_first
        }

    @staticmethod
    def pool_first(x):
        # x: [N x K x D]
        # return [N x D]
        return x[:, 0]

    @staticmethod
    def pool_avg(x):
        # x: [N x K x D]
        # return [N x D]
        return x.mean(dim=1)

    def forward(self, x):
        if self.pool:
            x = self.pool_fns[self.pool](x)
            x = self.transform_act_fn(x)
            x = self.linear(x)
        else:
            n, k, *_ = x.shape
            x = self.transform_act_fn(x)
            x = self.linear(x.view(n * k, *_)).view(n, k, -1)
        return x


class VideoTransformer(BertPreTrainedModel):
    def __init__(self, cfg, args, tok):
        super().__init__(cfg)
        self.embeddings = VideoTransformerEmbedder(cfg, args, tok)
        self.encoder = BertEncoder(cfg)
        self.init_weights()
        self.args = args
        self.tok = tok
        self.cfg = cfg

    def forward(self, x, lens, clip_ys, idxs, video_ts, _clip_embs=None):
        x, lens, clip_ys, nsp_ys, clip_embs, idxs, video_ts = self.embeddings(x, lens, clip_ys, idxs=idxs,
                                                                              video_ts=video_ts,
                                                                              _clip_embs=_clip_embs)

        attention_mask = torch.arange(max(lens))[None, :] < lens[:, None]
        attention_mask = attention_mask[:, None, None, :]
        attention_mask = attention_mask.to(dtype=x.dtype, device=x.device)
        attention_mask = (1.0 - attention_mask) * -10000.0

        x = self.encoder(x, attention_mask=attention_mask, head_mask=[None] * self.cfg.num_hidden_layers)
        # x: [N x K x HIDDEN_DIM]
        return x, lens, clip_ys, nsp_ys, clip_embs, idxs, video_ts


class AhaModel(nn.Module):
    def __init__(self, args, tok=None):
        super().__init__()
        cfg = BertConfig.from_json_file(args.config_path)
        cfg.hidden_size = args.hidden_dim
        cfg.vocab_size = 3  # [SEP], [CLS], [PAD]
        cfg.type_vocab_size = 3  # seq 0 vid, seq 1 vid, text
        self.video_transformer = VideoTransformer(cfg, args, tok)
        self.clip_prediction = VideoTransformerHead(d_in=args.hidden_dim, d_out=3, hidden_act=cfg.hidden_act)
        self.next_seq_prediction = VideoTransformerHead(d_in=args.hidden_dim, d_out=2, hidden_act=cfg.hidden_act,
                                                        pool='first')
        self.args = args
        if self.args.svo:
            self.svo_decoder_head = nn.Sequential(nn.GELU(), nn.Linear(args.hidden_dim, args.svo_dim * 3))
            self.svo_decoder_embs = nn.Linear(args.svo_dim, args.svo_vocab_size, bias=False)
            if self.args.svo_pretrained_embs:
                tok = BertTokenizer.from_pretrained('bert-base-uncased')
                bert = BertModel.from_pretrained('bert-base-uncased')
                self.svo_decoder_embs.weight.data = torch.stack([bert.embeddings.word_embeddings(
                    torch.tensor(tok.encode(_, add_special_tokens=False) if _ else [0])).mean(dim=0) for _ in
                                                                    args.svo_vocab])
                del bert
                del tok

    def forward(self, x, lens, clip_ys, idxs=None, video_ts=None, clip_loss_weights=None, clip_loss_sample_probs=None,
                _clip_embs=None, _xfmr_tuple=None, override_no_svo=False,
                svo_ys=None, svo_eval=False, pre_svo_reprs=None):
        lens_ = lens

        fwd_svo = self.args.svo and not override_no_svo

        if pre_svo_reprs is not None:
            svo_reprs = self.svo_decoder_head(pre_svo_reprs).view(-1, self.args.svo_dim)
            return (*self.svo_calc(svo_reprs, svo_ys, svo_eval, lens_, clip_ys), x)

        if self.args.load_cvpr_model and not fwd_svo:
            if _clip_embs is not None:
                x = _clip_embs
            else:
                x = self.cnn(x)
            x_cnn = x
            x = self.cvpr_oops_decoder(x)
            loss = F.cross_entropy(x, clip_ys, ignore_index=-1)
            corr = (x.argmax(dim=1) == clip_ys)[clip_ys != -1]
            acc = 100 * (corr.sum().item() / (len(corr) or 1))
            return loss, acc, x_cnn, x

        if self.args.svo_cnn_only:
            if _clip_embs is not None:
                x = _clip_embs
            else:
                x = self.cnn(x)
            x = x.detach()
            svo_reprs = self.svo_decoder_head(x).view(-1, self.args.svo_dim)
            return (*self.svo_calc(svo_reprs, svo_ys, svo_eval, lens_, clip_ys), x)

        if _xfmr_tuple is not None:
            clip_embs = None
            x, lens, clip_ys, nsp_ys = _xfmr_tuple
            if fwd_svo:
                valid_idxs = (clip_ys != -1).view(-1, max(lens))
                # valid_idxs = clip_ys != -1
        else:
            try:
                _clip_embs = _clip_embs.split(lens)
            except:
                pass
            (x,), lens, clip_ys, nsp_ys, clip_embs, idxs, video_ts = self.video_transformer(x, lens, clip_ys,
                                                                                            _clip_embs=_clip_embs,
                                                                                            video_ts=video_ts,
                                                                                            idxs=idxs)
            if fwd_svo:
                valid_idxs = clip_ys != -1
            clip_ys[nsp_ys == 0] = -1  # Don't predict clip labels if data doesn't follow in NSP
            clip_ys_ = clip_ys
            clip_ys = clip_ys.view(-1)
            if clip_loss_sample_probs is not None:
                clip_y_sample_ps = torch.zeros_like(clip_ys).to(torch.float)
                for y, p_sample_y in enumerate(clip_loss_sample_probs):
                    clip_y_sample_ps[clip_ys == y] = p_sample_y
                clip_y_sample_ps = torch.bernoulli(clip_y_sample_ps).bool()
                clip_ys[~clip_y_sample_ps] = -1

        if fwd_svo:
            # [(N x K x (3 * SVO_DIM)] before mask -> [N' x (3 * SVO_DIM)] -> [(N' * 3) x SVO_DIM]
            x = x.detach()[valid_idxs]
            svo_reprs = self.svo_decoder_head(x).view(-1, self.args.svo_dim)
            return (*self.svo_calc(svo_reprs, svo_ys, svo_eval, lens_, clip_ys), x)

        clip_y_preds = self.clip_prediction(x)  # [N x K x 3]
        clip_y_preds = clip_y_preds.view(-1, clip_y_preds.shape[-1])
        nsp_y_preds = self.next_seq_prediction(x)  # [N x 2]

        clip_loss = F.cross_entropy(clip_y_preds, clip_ys, ignore_index=-1, weight=clip_loss_weights,
                                    reduction='sum') / (sum(clip_ys != -1) or 1)
        nsp_loss = F.cross_entropy(nsp_y_preds, nsp_ys, ignore_index=-1, reduction='sum') / (sum(nsp_ys != -1) or 1)

        clip_correct = (clip_y_preds.argmax(dim=1) == clip_ys)[clip_ys != -1]
        clip_acc = 100 * clip_correct.sum().item() / (len(clip_correct) or 1)

        nsp_correct = (nsp_y_preds.argmax(dim=1) == nsp_ys)[nsp_ys != -1]
        nsp_acc = 100 * nsp_correct.sum().item() / (len(nsp_correct) or 1)

        try:
            split_x = x[clip_ys_ != -1].cpu().split(lens_.tolist())
        except Exception as e:
            split_x = None
            clip_embs = None

        return (clip_loss, nsp_loss, clip_acc, nsp_acc, clip_y_preds, nsp_y_preds, clip_ys, nsp_ys, lens, clip_embs, \
            split_x, x, idxs, video_ts)

    def svo_calc(self, svo_reprs, svo_ys, svo_eval=False, lens=None, clip_ys=None, eps=1e-4, max_pool=True):
        svo_scores = self.svo_decoder_embs(svo_reprs)  # [(N' * 3) x |SVO_VOCAB|]
        if svo_eval:
            ret = {}
            vocab_size = svo_scores.shape[1]
            clip_ys = clip_ys[clip_ys != -1]
            split_clip_ys = clip_ys.split(lens.tolist())
            bisected_lens = [_ for _ in list(itertools.chain.from_iterable(
                [(gt_index(ys), len(ys) - gt_index(ys)) for ys in
                    split_clip_ys])) if _ > 0]
            svo_scores = svo_scores.view(-1, 3, vocab_size).split(bisected_lens)
            svo_scores = torch.stack([scores.mean(dim=0) for scores in svo_scores])
            # N'' is number of bisected lens (should be batch size * 2)
            svo_scores = svo_scores.view(-1, vocab_size)  # [(N'' * 3) x |V|]
            svo_scores_reshaped = svo_scores.view(-1, 3, vocab_size)
            try:
                svo_ys = torch.stack([_[:, 0] for _ in svo_ys.split(bisected_lens, dim=1)]).transpose(0, 1)
            except:
                try:
                    svo_ys = torch.stack([_[0] for _ in svo_ys.split(bisected_lens, dim=0)]).transpose(0, 1)
                except:
                    if self.args.local_rank <= 0:
                        import ipdb
                        ipdb.set_trace()

            try:
                svo_ys = svo_ys.reshape(len(svo_ys), -1)  # [K_SVO x (N'' * 3)]
            except:
                if self.args.local_rank <= 0:
                    import ipdb
                    ipdb.set_trace()
            svo_ys_reshaped = svo_ys.view(len(svo_ys), -1, 3)
            svo_loss = torch.zeros((len(svo_scores_reshaped),), device=svo_ys.device, dtype=torch.float)
            svo_valid_ct = torch.zeros((len(svo_scores_reshaped),), device=svo_ys.device, dtype=torch.int)
            svo_loss_per = torch.zeros((len(svo_scores),), device=svo_ys.device, dtype=torch.float)
            svo_valid_ct_per = torch.zeros((len(svo_scores),), device=svo_ys.device, dtype=torch.int)
            for svo_ys_ in svo_ys:
                svo_loss_ = F.cross_entropy(svo_scores, svo_ys_, ignore_index=-1, reduction='none')
                m = (svo_ys_ != -1)
                svo_loss_per[m] += svo_loss_[m]
                svo_valid_ct_per[m] += 1
                svo_loss_ = svo_loss_.view(-1, 3).mean(dim=-1)
                m = (svo_ys_.view(-1, 3) != -1).all(dim=1)
                # svo_loss[svo_ys_ != 1] = torch.min(svo_loss_[svo_ys_ != 1], svo_loss[svo_ys_ != 1])
                # svo_loss = torch.min(svo_loss_, svo_loss)
                svo_loss[m] += svo_loss_[m]
                svo_valid_ct[m] += 1
            svo_loss /= (svo_valid_ct + eps)
            svo_loss = svo_loss[svo_valid_ct != 0]
            svo_loss_per /= (svo_valid_ct_per + eps)
            svo_loss_per = svo_loss_per[svo_valid_ct_per != 0]
            ret['svo_loss'] = svo_loss
            for i, pos in enumerate(('subj', 'verb', 'obj')):
                ret[f'svo_loss_{pos}'] = svo_loss_per[i::3]
            for k in (1, 5):
                _, scores_top_k = svo_scores.topk(k=k)  # [(N'' * 3) x k]
                ret[f'svo_preds_top{k}'] = scores_top_k.view(-1, 3, k)
                n_preds = len(scores_top_k)
                # below are [N'']
                corr_top_k = torch.zeros((len(svo_scores_reshaped),), device=svo_ys.device)
                corr_top_k_allthree = torch.zeros((len(svo_scores_reshaped),), device=svo_ys.device)
                corr_top_k_per = torch.zeros((len(svo_scores),), device=svo_ys.device)
                for svo_ys_ in svo_ys:
                    m = (svo_ys_ != -1)
                    svo_ys_ = svo_ys_.unsqueeze(-1)
                    corr_top_k_ = (scores_top_k == svo_ys_).any(dim=-1).to(int)
                    corr_top_k_per[m] += corr_top_k_[m]

                    # below will be [N''] - how many of each group's S, V, O predicted within top k?
                    corr_top_k_ = (scores_top_k == svo_ys_).view(-1, 3, k).any(dim=-1).sum(dim=-1)
                    corr_top_k_allthree_ = (corr_top_k_ == 3)  # [N'']
                    m = (svo_ys_.view(-1, 3) != -1).all(dim=1)
                    corr_top_k[m] += corr_top_k_[m]
                    corr_top_k_allthree[m] += corr_top_k_allthree_[m]
                if max_pool:
                    corr_top_k.clamp_max_(1)
                    corr_top_k_allthree.clamp_max_(1)
                    corr_top_k_per.clamp_max_(1)
                    svo_valid_ct.clamp_max_(1)
                    svo_valid_ct_per.clamp_max_(1)
                corr_top_k /= (svo_valid_ct + eps)
                corr_top_k_allthree /= (svo_valid_ct + eps)
                corr_top_k_per = corr_top_k_per.view(-1, 3).sum(dim=0)
                svo_valid_ct_per__ = svo_valid_ct_per.view(-1, 3).sum(dim=0)
                corr_top_k_per /= (svo_valid_ct_per__ + eps)
                acc_top_k = 100 * (corr_top_k.sum().item() / n_preds)
                acc_top_k_per = 100 * corr_top_k_per
                ret[f'svo_pooled_acc_{k}'] = acc_top_k
                acc_top_k_allthree = 100 * (corr_top_k_allthree.sum().item() / len(corr_top_k_allthree))
                ret[f'svo_pooled_acc_allthree_{k}'] = acc_top_k_allthree
                for i, pos in enumerate(('subj', 'verb', 'obj')):
                    ret[f'svo_pooled_acc_{pos}_{k}'] = acc_top_k_per[i]
            if self.args.svo_compute_rank:
                log_svo_scores = svo_scores.log_softmax(dim=-1)
                log_svo_scores_reshaped = log_svo_scores.view(-1, 3, vocab_size)
                log_p_svos = log_svo_scores.gather(dim=-1, index=svo_ys.clamp(min=0).t()).t().view(len(svo_ys), -1,
                                                                                                   3).sum(
                    dim=-1)
                log_p_svos[(svo_ys_reshaped == -1).all(dim=-1)] = -math.inf
                # above is [K_SVO x N'']
                n_smaller_svo = torch.zeros_like(svo_ys_reshaped[..., 0])
                for svo_i, row in enumerate(n_smaller_svo):
                    for batch_i, cell in enumerate(row):
                        log_p_tgt_svo = log_p_svos[svo_i, batch_i]
                        if log_p_tgt_svo == -math.inf:
                            continue
                        log_p_s_vec, log_p_v_vec, log_p_o_vec = log_svo_scores_reshaped[batch_i]
                        s_cond = log_p_s_vec < log_p_tgt_svo
                        cell += s_cond.sum() * (vocab_size ** 2)
                        for log_p_s in log_p_s_vec[~s_cond]:
                            log_p_sv_vec = log_p_s + log_p_v_vec
                            sv_cond = log_p_sv_vec < log_p_tgt_svo
                            cell += sv_cond.sum() * vocab_size
                            cell += ((log_p_sv_vec[~sv_cond][None] + log_p_o_vec[..., None]) < log_p_tgt_svo).sum()
                rank_svo = (vocab_size ** 3 - n_smaller_svo).to(float) / (vocab_size ** 3)
                rank_svo[(svo_ys_reshaped == -1).all(dim=-1)] = 0
                rank_svo = rank_svo.sum(dim=0) / ((svo_ys_reshaped != -1).all(dim=-1).sum(dim=0) + 0.0001)
                ret['svo_rank'] = rank_svo
            return ret,
        else:
            svo_ys = svo_ys.view(-1)  # [N' x 3] -> [(N' * 3)]
            svo_loss = F.cross_entropy(svo_scores, svo_ys, ignore_index=-1)
            svo_correct = (svo_scores.argmax(dim=1) == svo_ys)
            svo_acc = 100 * svo_correct[svo_ys != -1].sum().item() / (len(svo_correct[svo_ys != -1]) or 1)
            svo_valid_allthree = (svo_ys != -1).view(-1, 3).min(dim=1).values
            svo_correct_allthree = svo_correct.view(-1, 3).min(dim=1).values
            svo_acc_allthree = 100 * svo_correct_allthree[svo_valid_allthree].sum().item() / \
                               (len(svo_correct_allthree[svo_valid_allthree]) or 1)
            return svo_loss, svo_acc, svo_acc_allthree
