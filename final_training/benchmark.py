"""Bounded disposable benchmark, never saves learned weights or thesis metrics."""
import argparse
import gc
import json
import math
import statistics
import time

import torch
from core import (PROJECT, config, seed_all, model, configure_stage, training_mode,
                  loader, train_weights, autocast, optimization_loss, precision)


def run_case(cfg, frozen, warmup, steps):
    seed_all(cfg['seed'])
    net = model(cfg).cuda()
    teacher = model(config('M4')).cuda().eval() if cfg['id'] == 'M5' else None
    if teacher:
        teacher.requires_grad_(False)
    optimizer = configure_stage(net, cfg, frozen)
    training_mode(net, cfg, frozen)
    _, weights = train_weights(cfg)
    weights = weights.cuda()
    scaler = torch.amp.GradScaler('cuda', enabled=precision() == torch.float16)
    batches = loader(cfg, 'train', cfg['id'] == 'M5')
    iterator = iter(batches)
    durations = []
    acceptance = []
    torch.cuda.reset_peak_memory_stats()
    start_all = time.perf_counter()
    for index in range(warmup + steps):
        torch.cuda.synchronize()
        start = time.perf_counter()
        batch = tuple(t.cuda(non_blocking=True) for t in next(iterator))
        optimizer.zero_grad(set_to_none=True)
        with autocast():
            loss, stats = optimization_loss(net, batch, weights, cfg, teacher)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite benchmark objective')
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize()
        duration = time.perf_counter() - start
        if index >= warmup:
            durations.append(duration)
            if stats:
                acceptance.append(float(stats['gate_acceptance']))
    peak_allocated = torch.cuda.max_memory_allocated() / 2**30
    peak_reserved = torch.cuda.max_memory_reserved() / 2**30
    # Evaluate on training images only to time inference without reading validation.
    net.eval()
    evaluation_times = []
    for _ in range(steps):
        torch.cuda.synchronize()
        start = time.perf_counter()
        batch = next(iterator)
        images = batch[0].cuda(non_blocking=True)
        with torch.no_grad(), autocast():
            logits = net(images)
            _ = torch.nn.functional.cross_entropy(logits.float(), batch[-1].cuda(non_blocking=True))
        torch.cuda.synchronize()
        evaluation_times.append(time.perf_counter() - start)
    output = {'model': cfg['id'], 'stage': 'head' if frozen else 'finetune', 'batch_size': cfg['batch_size'],
              'warmup_updates': warmup, 'measured_updates': steps, 'measured_evaluation_batches': steps,
              'mean_step_seconds': statistics.mean(durations), 'median_step_seconds': statistics.median(durations),
              'p90_step_seconds': sorted(durations)[max(0, math.ceil(.9 * len(durations)) - 1)],
              'mean_evaluation_batch_seconds': statistics.mean(evaluation_times),
              'images_per_second': cfg['batch_size'] / statistics.mean(durations),
              'peak_allocated_gib': peak_allocated, 'peak_reserved_gib': peak_reserved,
              'wall_seconds': time.perf_counter() - start_all,
              'precision': str(precision()), 'initialization': 'random; disposable compute check only',
              'real_train_images_only': True, 'learned_checkpoint_saved': False,
              'gate_acceptance_benchmark_only': statistics.mean(acceptance) if acceptance else None,
              'timing_includes': 'train-only image loading/augmentation, H2D, forward, backward, AdamW; first loader startup excluded by warmup'}
    del net, teacher, optimizer, batches, iterator, batch, loss, logits
    gc.collect()
    torch.cuda.empty_cache()
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--warmup', type=int, default=5)
    args = parser.parse_args()
    if not 5 <= args.steps <= 40 or not 1 <= args.warmup <= 10:
        parser.error('Bounded benchmark requires 5–40 measured updates and 1–10 warmups')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; run authorized GPU check outside the sandbox')
    torch.set_num_threads(4)
    free, total = torch.cuda.mem_get_info()
    report = {'gpu': torch.cuda.get_device_name(0), 'cuda_available': True,
              'free_gib_at_start': free / 2**30, 'total_gib': total / 2**30,
              'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
              'cases': [], 'estimation_limitations': [
                  'Short warm-cache sample with random weights, not an epoch or performance experiment.',
                  'Extrapolation uses current provisional split; remediation can change counts.',
                  'Validation timing uses training-image inference, not held-out images or labels.',
                  'Intervals add 25–100% for startup/checkpoints/desktop load and throughput variability.',
                  'M5 random teacher gate is not a quality estimate; both teacher views are always computed.']}
    for key in ['M1', 'M2', 'M3', 'M4', 'M5']:
        cfg = config(key)
        for frozen in ([True, False] if key != 'M5' else [False]):
            failures = []
            for batch_size in [cfg['batch_size'], *cfg['batch_size_fallbacks']]:
                cfg['batch_size'] = batch_size
                try:
                    case = run_case(cfg, frozen, args.warmup, args.steps)
                    case['oom_attempts'] = failures
                    report['cases'].append(case)
                    print(json.dumps(case), flush=True)
                    break
                except torch.cuda.OutOfMemoryError:
                    failures.append(batch_size)
                    gc.collect()
                    torch.cuda.empty_cache()
            else:
                raise RuntimeError(f'All bounded memory fallbacks exhausted: {key}')
        (PROJECT / 'reports/benchmark.json').write_text(json.dumps(report, indent=2) + '\n')
    estimates = {}
    for key in ['M1', 'M2', 'M3', 'M4', 'M5']:
        cases = [c for c in report['cases'] if c['model'] == key]
        cfg = config(key)
        values = {}
        for case in cases:
            b = case['batch_size']
            values[case['stage']] = math.ceil(13686 / b) * case['mean_step_seconds'] + math.ceil(2835 / b) * case['mean_evaluation_batch_seconds']
        nominal = values.get('head', 0) * cfg['frozen_epochs'] + values['finetune'] * (cfg['max_total_epochs'] - cfg['frozen_epochs'])
        estimates[key] = {'seconds_per_epoch_including_validation': values, 'max_budget_nominal_minutes': nominal / 60,
                          'planning_interval_minutes': [nominal / 60 * 1.25, nominal / 60 * 2],
                          'max_epochs': cfg['max_total_epochs'], 'budget_kind': cfg['budget_kind']}
    estimates['budget_accounting'] = {'M5': '40 ADDITIONAL epochs after up to 30 M3 epochs; frozen M4 teacher requires up to 30 epochs too. All M5 weak/strong student and teacher forwards are timed.',
                                    'early_stopping': 'Actual runs may end earlier; no minimum duration guaranteed.'}
    report['runtime_estimates'] = estimates
    (PROJECT / 'reports/benchmark.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(estimates, indent=2))


if __name__ == '__main__':
    main()
