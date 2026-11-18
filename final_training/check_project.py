"""Contract checks only: no optimization loop and no checkpoint production."""
import json
from pathlib import Path

import torch
from core import PROJECT, ROOT, CLASSES, config, train_weights, rows_for, m5_loss
from train import approval_check, load_final_parent


def main():
    checks = {}
    assert sorted(path.stem for path in (PROJECT / 'configs').glob('*.json')) == ['M1', 'M2', 'M3', 'M4', 'M5']
    checks['exactly_five_main_configs'] = True
    for key in ['M1', 'M2', 'M3', 'M4', 'M5']:
        cfg = config(key)
        assert cfg['input_size'] == [3, 224, 224] and cfg['classes'] == CLASSES
        assert cfg['optimizer'] == 'AdamW' and cfg['weight_decay'] == 1e-4
        assert cfg['head_lr'] == 1e-4 and cfg['backbone_lr'] == 1e-5
        assert cfg['frozen_head_lr'] == 1e-3 and cfg['early_stopping']['patience'] == 6
        assert cfg['max_total_epochs'] == (40 if key == 'M5' else 30)
        assert cfg['frozen_epochs'] == (0 if key == 'M5' else 5)
        assert (ROOT / cfg['checkpoint_dir']).is_dir()
    checks['requested_hyperparameters_and_directories'] = True
    cfg = config('M5')
    train = rows_for(cfg, 'train')
    validation = rows_for(cfg, 'validation')
    assert not {r['image'] for r in train} & {r['image'] for r in validation}
    counts, weights = train_weights(cfg)
    assert counts == [7956, 3011, 2293, 426]
    checks['train_only_counts'] = counts
    checks['train_only_ce_weights'] = weights.tolist()
    for forbidden in ['test', 'external']:
        try:
            rows_for(cfg, forbidden)
        except ValueError:
            pass
        else:
            raise AssertionError('Forbidden optimization split accepted')
    checks['test_external_loaders_rejected'] = True
    try:
        approval_check(cfg)
    except RuntimeError as exc:
        assert 'leakage' in str(exc)
    else:
        raise AssertionError('Unresolved data gate was not enforced')
    checks['full_training_blocked_by_unresolved_leakage'] = True
    try:
        load_final_parent('experiments/demo_checkpoint.pt', 'M3', {})
    except ValueError:
        pass
    else:
        raise AssertionError('Demonstration checkpoint accepted')
    checks['nonfinal_parent_paths_rejected'] = True
    torch.manual_seed(42)
    sw, ss = torch.randn(4, 4, requires_grad=True), torch.randn(4, 4, requires_grad=True)
    target = torch.arange(4)
    teacher = (torch.eye(4) * 10).requires_grad_()
    loss, stats = m5_loss(sw, ss, teacher, teacher, target, weights, cfg)
    assert stats['gate_acceptance'] == 1 and torch.isfinite(loss)
    loss.backward()
    assert sw.grad is not None and ss.grad is not None and teacher.grad is None
    uniform = torch.zeros(4, 4)
    loss, stats = m5_loss(sw.detach(), ss.detach(), uniform, uniform, target, weights, cfg)
    assert stats['gate_acceptance'] == 0 and stats['kd'] == 0 and stats['consistency'] == 0
    assert torch.allclose(loss, stats['ce'] * cfg['loss']['lambda_ce'])
    checks['m5_teacher_detached_student_gradients_and_zero_gate_fallback'] = True
    assert not list((PROJECT / 'runs').rglob('*.pt'))
    checks['no_final_trained_checkpoints_exist'] = True
    (PROJECT / 'reports/contract_checks.json').write_text(json.dumps(checks, indent=2) + '\n')
    print(json.dumps(checks, indent=2))


if __name__ == '__main__':
    main()
