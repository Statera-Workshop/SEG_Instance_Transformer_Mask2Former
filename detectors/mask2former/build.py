"""
Mask2Former (ResNet-50, MSDeformAttn pixel decoder, 9-layer masked-attention decoder, 100 queries) builder. Seg task.

Paper: Cheng, Misra, Schwing, Kirillov & Girdhar, "Masked-attention Mask Transformer for Universal Image
       Segmentation" (CVPR 2022, arXiv 2112.01527).
Code : facebookresearch/Mask2Former (MIT), configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml;
       re-written here following, line by line:
         modeling/pixel_decoder/msdeformattn.py   MSDeformAttnPixelDecoder: res3-res5 -> 1x1 conv + GN(32), sine
             position + level embedding, 6 MSDeformAttn encoder layers (256, 8 heads, 4 points, FFN 1024, dropout 0),
             then FPN on res2 (1x1 lateral + GN, + bilinear-upsampled res3 output, 3x3 conv + GN + ReLU) and a 1x1
             mask_features conv -> per-pixel embeddings at stride 4 (mask_dim 256); multi-scale features res5, res4, res3
             (the encoder layers are the Deformable-DETR ones Mask2Former copied: detectors/deformable_detr/official)
         modeling/transformer_decoder/mask2former_transformer_decoder.py   MultiScaleMaskedTransformerDecoder:
             100 learnable query features + query positions, 9 layers (DEC_LAYERS 10 - 1) of masked cross-attention ->
             self-attention -> FFN 2048 (post-norm, dropout 0), round-robin over res5 / res4 / res3 with level
             embeddings; attention mask = previous prediction resized to the level, sigmoid < 0.5 (a query whose mask
             is empty attends everywhere); class Linear(C + 1), mask MLP 3 x 256, mask = embed . per-pixel embeddings;
             predictions before the first layer and after every layer (10 sets, deep supervision)
         modeling/matcher.py   Hungarian matching on 12,544 uniformly sampled points: cost class 2 (-p), mask 5
             (sigmoid CE), dice 5
         modeling/criterion.py  CE with no-object weight 0.1 (x2); mask BCE (x5) and dice (x5) on 12,544 points per
             mask chosen by uncertainty (oversample 3, importance 0.75), normalised by the number of gt masks; the
             same losses on every auxiliary prediction (point_sample = detectron2 point_features)
         maskformer_model.py instance_inference: class softmax x mask score (mean sigmoid inside the mask > 0),
             boxes = tight boxes of the masks (no box head)

Raw contract: one row per query: tight box of mask logits > 0 (stride 4, input px), objectness = mask score, class
probabilities = softmax without the no-object class; aux = mask logits [B, 100, S/4, S/4]; decode_masks kind "full".

[Unified] as detectors/detr / deformable_detr: no ImageNet weights, trainable BN (FrozenBN in the original) +
PreciseBN on the backbone, common init of Conv2d / BN (transformer Linear layers keep the official xavier init),
fixed 608 input without padding masks, Adam 1e-4 for all parameters instead of AdamW 1e-4 / backbone x0.1 / weight
decay 0.05 (the official full-model gradient clipping 0.01 is kept, as DETR's 0.1), 30 epochs instead of 50,
common augmentation; the common post-processing (score = class x mask score per query, top-300, class-wise NMS 0.5,
max 100) replaces the official top-100 over queries x classes without NMS.
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from detectors.anchor_free.common import HeadStats, to_raw
from detectors.deformable_detr.official.deformable_transformer import (DeformableTransformerEncoder,
                                                                       DeformableTransformerEncoderLayer)
from detectors.deformable_detr.official.ms_deform_attn import MSDeformAttn
from detectors.detr.official.misc_min import NestedTensor
from detectors.detr.official.position_encoding import PositionEmbeddingSine
from detectors.retinanet.resnet import build_resnet
from detectors.yolo11_seg.build import split_seg_targets

NUM_QUERIES, HIDDEN, MASK_DIM, DEC_LAYERS, NUM_LEVELS = 100, 256, 256, 9, 3
CLASS_W, MASK_W, DICE_W, EOS_COEF = 2.0, 5.0, 5.0, 0.1
TRAIN_POINTS, OVERSAMPLE, IMPORTANCE = 12544, 3.0, 0.75


# ----------------------------------------------------------------------------- point sampling (detectron2)
def point_sample(inp, coords):
    """inp [N, 1, H, W], coords [N, P, 2] in [0, 1] (x, y) -> [N, 1, P] (bilinear, align_corners False)."""
    return F.grid_sample(inp, 2.0 * coords.unsqueeze(2) - 1.0, align_corners=False).squeeze(3)


@torch.no_grad()
def uncertain_points(logits, num_points=TRAIN_POINTS, oversample=OVERSAMPLE, importance=IMPORTANCE):
    """get_uncertain_point_coords_with_randomness with uncertainty = -|logit|. logits [N, 1, H, W]."""
    n = logits.shape[0]
    n_sampled = int(num_points * oversample)
    coords = torch.rand(n, n_sampled, 2, device=logits.device)
    unc = -point_sample(logits, coords).abs()                                  # [N, 1, n_sampled]
    n_unc = int(importance * num_points)
    idx = unc[:, 0].topk(n_unc, dim=1).indices + n_sampled * torch.arange(n, device=logits.device)[:, None]
    coords = coords.view(-1, 2)[idx.view(-1)].view(n, n_unc, 2)
    if num_points - n_unc > 0:
        coords = torch.cat([coords, torch.rand(n, num_points - n_unc, 2, device=logits.device)], 1)
    return coords


def dice_loss(inputs, targets, num_masks):
    p = inputs.sigmoid()
    num = 2 * (p * targets).sum(-1)
    den = p.sum(-1) + targets.sum(-1)
    return (1 - (num + 1) / (den + 1)).sum() / num_masks


def sigmoid_ce_loss(inputs, targets, num_masks):
    return F.binary_cross_entropy_with_logits(inputs, targets, reduction="none").mean(1).sum() / num_masks


def batch_dice(inputs, targets):
    p = inputs.sigmoid()
    num = 2 * torch.einsum("nc,mc->nm", p, targets)
    return 1 - (num + 1) / (p.sum(-1)[:, None] + targets.sum(-1)[None, :] + 1)


def batch_sigmoid_ce(inputs, targets):
    hw = inputs.shape[1]
    pos = F.binary_cross_entropy_with_logits(inputs, torch.ones_like(inputs), reduction="none")
    neg = F.binary_cross_entropy_with_logits(inputs, torch.zeros_like(inputs), reduction="none")
    return (torch.einsum("nc,mc->nm", pos, targets) + torch.einsum("nc,mc->nm", neg, 1 - targets)) / hw


@torch.no_grad()
def hungarian_match(logits, masks, tgt):
    """modeling/matcher.py HungarianMatcher.memory_efficient_forward. logits [B, Q, C+1], masks [B, Q, h, w]."""
    out = []
    for b, t in enumerate(tgt):
        if len(t["labels"]) == 0:
            out.append((torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long)))
            continue
        prob = logits[b].softmax(-1)
        cost_class = -prob[:, t["labels"]]
        coords = torch.rand(1, TRAIN_POINTS, 2, device=masks.device)
        tm = point_sample(t["masks"][:, None].float(), coords.repeat(len(t["labels"]), 1, 1))[:, 0]
        om = point_sample(masks[b][:, None], coords.repeat(masks.shape[1], 1, 1))[:, 0]
        cost = MASK_W * batch_sigmoid_ce(om.float(), tm.float()) + CLASS_W * cost_class + \
            DICE_W * batch_dice(om.float(), tm.float())
        i, j = linear_sum_assignment(cost.cpu().numpy())
        out.append((torch.as_tensor(i, dtype=torch.long), torch.as_tensor(j, dtype=torch.long)))
    return out


# ----------------------------------------------------------------------------- decoder layers (official)
class SelfAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=0.0)
        self.norm = nn.LayerNorm(d_model)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, tgt, query_pos):
        q = k = tgt + query_pos
        return self.norm(tgt + self.self_attn(q, k, value=tgt)[0])


class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=0.0)
        self.norm = nn.LayerNorm(d_model)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, tgt, memory, memory_mask, pos, query_pos):
        t2 = self.multihead_attn(query=tgt + query_pos, key=memory + pos, value=memory, attn_mask=memory_mask)[0]
        return self.norm(tgt + t2)


class FFNLayer(nn.Module):
    def __init__(self, d_model, dim_feedforward):
        super().__init__()
        self.linear1, self.linear2 = nn.Linear(d_model, dim_feedforward), nn.Linear(dim_feedforward, d_model)
        self.norm = nn.LayerNorm(d_model)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, tgt):
        return self.norm(tgt + self.linear2(F.relu(self.linear1(tgt))))


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < len(self.layers) - 1 else layer(x)
        return x


# ----------------------------------------------------------------------------- model
class Mask2Former(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = C = num_classes
        self.backbone = build_resnet(50)
        ch = {"res2": 256, **self.backbone.out_channels}
        self.pe = PositionEmbeddingSine(HIDDEN // 2, normalize=True)
        # pixel decoder
        self.input_proj = nn.ModuleList(nn.Sequential(nn.Conv2d(ch[f], HIDDEN, 1), nn.GroupNorm(32, HIDDEN))
                                        for f in ("res5", "res4", "res3"))
        layer = DeformableTransformerEncoderLayer(HIDDEN, 1024, 0.0, "relu", NUM_LEVELS, 8, 4)
        self.encoder = DeformableTransformerEncoder(layer, 6)
        self.pd_level_embed = nn.Parameter(torch.Tensor(NUM_LEVELS, HIDDEN))
        for p in self.encoder.parameters():                                      # MSDeformAttnTransformerEncoderOnly
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.encoder.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        nn.init.normal_(self.pd_level_embed)
        self.lateral = nn.Sequential(nn.Conv2d(ch["res2"], HIDDEN, 1, bias=False), nn.GroupNorm(32, HIDDEN))
        self.output_conv = nn.Sequential(nn.Conv2d(HIDDEN, HIDDEN, 3, 1, 1, bias=False), nn.GroupNorm(32, HIDDEN),
                                         nn.ReLU(inplace=True))
        self.mask_features = nn.Conv2d(HIDDEN, MASK_DIM, 1)
        # transformer decoder
        self.self_layers = nn.ModuleList(SelfAttentionLayer(HIDDEN, 8) for _ in range(DEC_LAYERS))
        self.cross_layers = nn.ModuleList(CrossAttentionLayer(HIDDEN, 8) for _ in range(DEC_LAYERS))
        self.ffn_layers = nn.ModuleList(FFNLayer(HIDDEN, 2048) for _ in range(DEC_LAYERS))
        self.decoder_norm = nn.LayerNorm(HIDDEN)
        self.query_feat = nn.Embedding(NUM_QUERIES, HIDDEN)
        self.query_embed = nn.Embedding(NUM_QUERIES, HIDDEN)
        self.level_embed = nn.Embedding(NUM_LEVELS, HIDDEN)
        self.class_embed = nn.Linear(HIDDEN, C + 1)
        self.mask_embed = MLP(HIDDEN, HIDDEN, MASK_DIM, 3)
        self.register_buffer("empty_weight", torch.cat([torch.ones(C), torch.tensor([EOS_COEF])]))
        # recipe
        self.precise_bn, self.precise_bn_images, self.precise_bn_batch_size = True, 1600, 16
        self.clip_max_norm = float(os.environ.get("M2F_CLIP", "0.01"))          # official full-model clipping
        self.stats = HeadStats()
        self.yolo_layers = [self.stats]

    def _backbone(self, x):
        bb = self.backbone
        x = bb.maxpool(bb.relu(bb.bn1(bb.conv1(x))))
        c2 = bb.layer1(x)
        c3 = bb.layer2(c2)
        c4 = bb.layer3(c3)
        return {"res2": c2, "res3": c3, "res4": c4, "res5": bb.layer4(c4)}

    def _pos(self, x):
        return self.pe(NestedTensor(x, torch.zeros(x.shape[0], *x.shape[-2:], dtype=torch.bool, device=x.device)))

    def pixel_decoder(self, feats):
        srcs, flat, lvl_pos, shapes = [], [], [], []
        for lvl, f in enumerate(("res5", "res4", "res3")):
            x = feats[f]
            src = self.input_proj[lvl](x)
            B, _, h, w = src.shape
            shapes.append((h, w))
            flat.append(src.flatten(2).transpose(1, 2))
            lvl_pos.append(self._pos(x).to(src.dtype).flatten(2).transpose(1, 2) + self.pd_level_embed[lvl].view(1, 1, -1))
        spatial = torch.as_tensor(shapes, dtype=torch.long, device=flat[0].device)
        start = torch.cat([spatial.new_zeros(1), spatial.prod(1).cumsum(0)[:-1]])
        valid = torch.ones(B, NUM_LEVELS, 2, device=flat[0].device)
        mem = self.encoder(torch.cat(flat, 1), spatial, start, valid, torch.cat(lvl_pos, 1), None)
        out = [y.transpose(1, 2).reshape(B, HIDDEN, h, w)
               for y, (h, w) in zip(mem.split([h * w for h, w in shapes], 1), shapes)]
        cur = self.lateral(feats["res2"])
        y = self.output_conv(cur + F.interpolate(out[-1], size=cur.shape[-2:], mode="bilinear", align_corners=False))
        return self.mask_features(y), out                                       # out: res5, res4, res3

    def _predict(self, output, mask_features, target_size):
        d = self.decoder_norm(output).transpose(0, 1)                           # [B, Q, D]
        cls = self.class_embed(d)
        masks = torch.einsum("bqc,bchw->bqhw", self.mask_embed(d), mask_features)
        am = F.interpolate(masks, size=target_size, mode="bilinear", align_corners=False)
        am = (am.sigmoid().flatten(2).unsqueeze(1).repeat(1, 8, 1, 1).flatten(0, 1) < 0.5).detach()
        return cls, masks, am

    def heads(self, x):
        mask_features, ms = self.pixel_decoder(self._backbone(x))
        src, pos, sizes = [], [], []
        for i in range(NUM_LEVELS):
            sizes.append(ms[i].shape[-2:])
            pos.append(self._pos(ms[i]).to(ms[i].dtype).flatten(2).permute(2, 0, 1))
            src.append((ms[i].flatten(2) + self.level_embed.weight[i][None, :, None]).permute(2, 0, 1))
        B = x.shape[0]
        query_pos = self.query_embed.weight.unsqueeze(1).repeat(1, B, 1)
        output = self.query_feat.weight.unsqueeze(1).repeat(1, B, 1)
        cls, masks, am = self._predict(output, mask_features, sizes[0])
        preds = [(cls, masks)]
        for i in range(DEC_LAYERS):
            lvl = i % NUM_LEVELS
            am[torch.where(am.sum(-1) == am.shape[-1])] = False
            output = self.cross_layers[i](output, src[lvl], am, pos[lvl], query_pos)
            output = self.self_layers[i](output, query_pos)
            output = self.ffn_layers[i](output)
            cls, masks, am = self._predict(output, mask_features, sizes[(i + 1) % NUM_LEVELS])
            preds.append((cls, masks))
        return preds

    def _set_loss(self, logits, masks, idx, tgt, num_masks):
        B, Q = logits.shape[:2]
        bi = torch.cat([torch.full_like(i, b) for b, (i, _) in enumerate(idx)]).to(logits.device)
        qi = torch.cat([i for i, _ in idx]).to(logits.device)
        tcls = torch.full((B, Q), self.num_classes, dtype=torch.long, device=logits.device)
        tcls[bi, qi] = torch.cat([t["labels"][j.to(logits.device)] for t, (_, j) in zip(tgt, idx)])
        loss_ce = F.cross_entropy(logits.transpose(1, 2), tcls, self.empty_weight)
        if len(qi) == 0:
            zero = masks.sum() * 0.0
            return loss_ce, zero, zero
        src = masks[bi, qi][:, None]
        tm = torch.cat([t["masks"][j.to(logits.device)] for t, (_, j) in zip(tgt, idx)]).float()[:, None]
        coords = uncertain_points(src)
        labels = point_sample(tm, coords)[:, 0]
        pl = point_sample(src, coords)[:, 0]
        return loss_ce, sigmoid_ce_loss(pl, labels, num_masks), dice_loss(pl, labels, num_masks)

    @staticmethod
    def _boxes_from_masks(masks, stride):
        """tight boxes of logits > 0, input px. masks [B, Q, h, w] -> [B, Q, 4]; empty masks -> 0 box."""
        m = masks > 0
        xs, ys = m.any(2), m.any(3)                                           # [B, Q, w], [B, Q, h]
        w, h = xs.shape[-1], ys.shape[-1]
        x1 = xs.float().argmax(-1)
        x2 = w - xs.flip(-1).float().argmax(-1)
        y1 = ys.float().argmax(-1)
        y2 = h - ys.flip(-1).float().argmax(-1)
        box = torch.stack([x1, y1, x2, y2], -1).float() * stride
        return box * xs.any(-1, keepdim=True)

    def forward(self, x, targets=None):
        B, C, S = x.shape[0], self.num_classes, x.shape[2]
        if targets is None and self.training:                                 # PreciseBN: BN only in the backbone
            self._backbone(x)
            return x.new_zeros(B, 0, 5 + C)
        preds = self.heads(x)
        logits, masks = preds[-1]
        if targets is None:
            pos = (masks > 0).flatten(2).float()
            mscore = (masks.sigmoid().flatten(2) * pos).sum(-1) / (pos.sum(-1) + 1e-6)
            box = self._boxes_from_masks(masks, S / masks.shape[-1])
            raw = to_raw(box, logits.softmax(-1)[..., :-1], mscore[..., None])
            return raw, {"masks": masks}
        tgt = split_seg_targets(targets, B, S)                                  # labels 0..C-1, masks [n, S, S]
        num_masks = max(sum(len(t["labels"]) for t in tgt), 1)
        loss_dict, loss, last_idx = {}, 0.0, None
        for k, (lg, mk) in enumerate(preds):
            idx = hungarian_match(lg.detach(), mk.detach(), tgt)
            ce, lm, ld = self._set_loss(lg, mk, idx, tgt, num_masks)
            loss = loss + CLASS_W * ce + MASK_W * lm + DICE_W * ld
            if k == len(preds) - 1:
                loss_dict.update(loss_ce=ce, loss_mask=lm, loss_dice=ld)
                last_idx = idx
        with torch.no_grad():                                                   # stats on the last decoder layer
            fg = torch.zeros(B, NUM_QUERIES, dtype=torch.bool, device=x.device)
            lab = torch.full((B, NUM_QUERIES), -1, dtype=torch.long, device=x.device)
            for b, (i, j) in enumerate(last_idx):
                fg[b, i.to(x.device)] = True
                lab[b, i.to(x.device)] = tgt[b]["labels"][j.to(x.device)]
            lg = logits.detach()
            prob = F.softmax(lg, -1)[..., :-1].reshape(-1, C)
        self.stats.update({"losses": loss, **loss_dict, "num_pos": fg.float().sum() / B},
                          prob.max(-1).values, lg[..., :-1].reshape(-1, C), fg.view(-1), lab.view(-1)[fg.view(-1)])
        return loss, x.new_zeros(B, 0, 5 + C)

    @torch.no_grad()
    def decode_masks(self, aux, b, rows, boxes, labels):
        return {"kind": "full", "prob": aux["masks"][b, rows].sigmoid()}


def build_model(task, num_classes):
    return Mask2Former(num_classes)
