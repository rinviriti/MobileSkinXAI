"""Shared final-only data, model, optimizer and candidate M5 objective."""
import csv
import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np
from PIL import Image
from safetensors.torch import load_file
import timm
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parents[2]
CLASSES = ['NV', 'MEL', 'BCC', 'SCC']
PROJECT_ID = 'MobileSkinXAI_final_v20261117T061722Z_f5055bd9'


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def config(model_id):
    result = json.loads((PROJECT / 'configs' / f'{model_id}.json').read_text())
    if result['id'] != model_id or result['project_id'] != PROJECT_ID or result['classes'] != CLASSES:
        raise ValueError('Unexpected final configuration identity')
    return result


def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(_):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def model(cfg, pretrained=False):
    # Benchmark/export calls use random initialization; final calls MUST load
    # a freshly instantiated official ImageNet artifact and replace its head.
    if not pretrained:
        return timm.create_model(cfg['architecture'], pretrained=False, num_classes=4)
    record_path = ROOT / 'final_training' / 'pretrained' / cfg['id'] / 'provenance.json'
    record = json.loads(record_path.read_text())
    artifact = record_path.parent / 'model.safetensors'
    if record['repo_id'] != cfg['initialization']['official_repo'] or digest(artifact) != record['sha256']:
        raise ValueError('Official initialization provenance/hash mismatch')
    result = timm.create_model(cfg['architecture'], pretrained=False, num_classes=1000)
    result.load_state_dict(load_file(str(artifact)), strict=True)
    result.reset_classifier(4)
    actual = timm.data.resolve_model_data_config(result)
    for key in ['mean', 'std']:
        if not np.allclose(actual[key], cfg['normalization'][key]):
            raise ValueError('Configuration does not match official pretrained normalization')
    return result


def configure_stage(net, cfg, frozen):
    for parameter in net.parameters():
        parameter.requires_grad_(not frozen)
    head = list(net.get_classifier().parameters())
    for parameter in head:
        parameter.requires_grad_(True)
    head_ids = {id(parameter) for parameter in head}
    groups = [{'params': head, 'lr': cfg['frozen_head_lr'] if frozen else cfg['head_lr']}]
    if not frozen:
        groups.append({'params': [p for p in net.parameters() if id(p) not in head_ids], 'lr': cfg['backbone_lr']})
    return torch.optim.AdamW(groups, weight_decay=cfg['weight_decay'])


def training_mode(net, cfg, frozen=False):
    net.train()
    if frozen:
        net.eval()  # Also disables frozen-backbone stochastic depth/dropout.
        net.get_classifier().train()
    if cfg['freeze_bn_running_stats']:
        for layer in net.modules():
            if isinstance(layer, nn.modules.batchnorm._BatchNorm):
                layer.eval()


def precision():
    if not torch.cuda.is_available():
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def autocast():
    return torch.autocast('cuda', dtype=precision(), enabled=torch.cuda.is_available())


def rows_for(cfg, split):
    if split not in ['train', 'validation']:
        raise ValueError('Trainer never loads test or PAD records')
    with (ROOT / cfg['split_manifest']).open(newline='') as handle:
        rows = [row for row in csv.DictReader(handle) if row['split'] == split]
    if not rows or len({row['image'] for row in rows}) != len(rows):
        raise ValueError('Empty or duplicate manifest partition')
    for row in rows:
        expected = f'{split}/{row["label"]}/{row["image"]}.jpg'
        if row['label'] not in CLASSES or int(row['label_index']) != CLASSES.index(row['label']) or row['processed_relative_path'] != expected:
            raise ValueError('Unexpected prepared label/path')
    return rows


def train_weights(cfg):
    rows = rows_for(cfg, 'train')
    counts = [sum(row['label'] == label for row in rows) for label in CLASSES]
    if min(counts) <= 0:
        raise ValueError('Missing training class')
    return counts, torch.tensor([len(rows) / (4 * n) for n in counts], dtype=torch.float32)


class Images(Dataset):
    def __init__(self, cfg, split, two_views=False, return_ids=False):
        if two_views and split != 'train':
            raise ValueError('Cross-view optimization is training-only')
        self.rows = rows_for(cfg, split)
        self.root = ROOT / cfg['data_root']
        self.train = split == 'train'
        self.two_views = two_views
        self.return_ids = return_ids
        augmentation = cfg['augmentation']
        geometry = augmentation['shared_geometry']
        self.geometry = T.Compose([T.RandomHorizontalFlip(geometry['horizontal_flip']),
                                   T.RandomVerticalFlip(geometry['vertical_flip']),
                                   T.RandomRotation(geometry['rotation_degrees'], interpolation=InterpolationMode.BILINEAR,
                                                    fill=(geometry['fill'],) * 3)])
        self.mild = T.ColorJitter(brightness=augmentation['baseline_color_jitter'],
                                  contrast=augmentation['baseline_color_jitter'], saturation=.05)
        self.strong = T.Compose([T.ColorJitter(*augmentation['strong_color_jitter']),
                                 T.RandomGrayscale(augmentation['strong_grayscale_probability'])])
        self.tensor = T.Compose([T.ToTensor(), T.Normalize(**cfg['normalization'])])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(self.root / row['processed_relative_path']) as source:
            source.load()
            if source.mode != 'RGB' or source.size != (224, 224):
                raise ValueError('Only verified 224x224 RGB inputs are accepted')
            image = source.copy()
        target = CLASSES.index(row['label'])
        if self.train:
            image = self.geometry(image)
            if self.two_views:
                result = (self.tensor(image), self.tensor(self.strong(image.copy())), target)
                return (*result, row['image']) if self.return_ids else result
            image = self.mild(image)
        result = (self.tensor(image), target)
        return (*result, row['image']) if self.return_ids else result


def loader(cfg, split, two_views=False, return_ids=False):
    generator = torch.Generator().manual_seed(cfg['seed'])
    return DataLoader(Images(cfg, split, two_views, return_ids), batch_size=cfg['batch_size'],
                      shuffle=split == 'train', num_workers=cfg['num_workers'], pin_memory=True,
                      persistent_workers=cfg['num_workers'] > 0, worker_init_fn=seed_worker,
                      generator=generator, drop_last=False)


def js_divergence(p, q, epsilon=1e-8):
    p, q = p.float().clamp_min(epsilon), q.float().clamp_min(epsilon)
    mid = (p + q) / 2
    return .5 * ((p * (p.log() - mid.log())).sum(-1) + (q * (q.log() - mid.log())).sum(-1))


def m5_loss(sw, ss, tw, ts, target, weights, cfg):
    """Teacher logits are detached; all weights/gates use training labels only."""
    c = cfg['loss']
    eps, temperature = c['epsilon'], c['temperature']
    sw, ss, tw, ts = [value.float() for value in [sw, ss, tw.detach(), ts.detach()]]
    p, q = tw.softmax(-1), ts.softmax(-1)
    entropy = -.5 * ((p * p.clamp_min(eps).log()).sum(-1) + (q * q.clamp_min(eps).log()).sum(-1)) / math.log(4)
    divergence = js_divergence(p, q, eps)
    confident = torch.minimum(p.max(-1).values, q.max(-1).values) >= c['confidence_threshold']
    gate = confident & (entropy <= c['max_normalized_entropy']) & (divergence <= c['max_teacher_js'])
    if c['require_teacher_correct_both_views']:
        gate &= (p.argmax(-1) == target) & (q.argmax(-1) == target)
    reliability = (gate.float() * (1 - entropy).clamp(0, 1) * torch.exp(-divergence / c['reliability_js_scale'])).detach()
    class_weight = weights[target].pow(c['class_weight_power'])
    weighted_gate = class_weight * reliability
    ce = .5 * (F.cross_entropy(sw, target, weight=weights) + F.cross_entropy(ss, target, weight=weights))
    # Cross-view KL: teacher weak -> student strong and vice versa.
    kd_per_image = .5 * temperature**2 * (
        F.kl_div(F.log_softmax(ss / temperature, -1), F.softmax(tw / temperature, -1), reduction='none').sum(-1) +
        F.kl_div(F.log_softmax(sw / temperature, -1), F.softmax(ts / temperature, -1), reduction='none').sum(-1))
    denominator = class_weight.sum().clamp_min(eps)
    kd = (weighted_gate * kd_per_image).sum() / denominator
    consistency = (weighted_gate * js_divergence(sw.softmax(-1), ss.softmax(-1), eps)).sum() / denominator
    total = c['lambda_ce'] * ce + c['lambda_kd'] * kd + c['lambda_consistency'] * consistency
    return total, {'ce': ce.detach(), 'kd': kd.detach(), 'consistency': consistency.detach(),
                   'gate_acceptance': gate.float().mean().detach(), 'gate_mass': reliability.mean().detach(),
                   'gate_mask': gate.detach()}


def optimization_loss(net, batch, weights, cfg, teacher=None, return_logits=False):
    if cfg['id'] != 'M5':
        images, target = batch
        logits = net(images)
        result = (F.cross_entropy(logits, target, weight=weights), {})
        return (*result, logits) if return_logits else result
    weak, strong, target = batch
    with torch.no_grad():
        tw, ts = teacher(weak), teacher(strong)
    weak_logits, strong_logits = net(weak), net(strong)
    result = m5_loss(weak_logits, strong_logits, tw, ts, target, weights, cfg)
    return (*result, weak_logits) if return_logits else result


def macro_f1(matrix):
    m = matrix.float()
    return (2 * m.diag() / (m.sum(0) + m.sum(1)).clamp_min(1)).mean().item()
