"""Guarded final trainer. Default invocation only inspects configuration.

Full execution requires --execute, a separately reviewed data approval, and
fresh official artifacts. This preparation session never uses --execute.
"""
import argparse
import os
import csv
from datetime import datetime, timezone
import json
import time
import uuid

import torch
from core import (PROJECT, ROOT, PROJECT_ID, config, digest, seed_all, model,
                  configure_stage, training_mode, precision, autocast, loader,
                  rows_for, train_weights, optimization_loss, macro_f1)


def approval_check(cfg):
    approval = json.loads((PROJECT / 'data_approval.json').read_text())
    if approval.get('approved') is not True:
        raise RuntimeError('Full training blocked: unresolved dataset leakage review; see data_approval.json')
    release = ROOT / cfg['dataset_release']
    if release.name != 'v20261117T060846Z_608314c8' or cfg['dataset_version'] != release.name:
        raise ValueError('Only the approved immutable release is permitted')
    lock = json.loads((release / 'RELEASE_LOCK.json').read_text())
    if digest(release / 'checksums.sha256') != lock['checksums_sha256']:
        raise ValueError('Release lock mismatch')
    for line in (release / 'checksums.sha256').read_text().splitlines():
        expected, name = line.split('  ', 1)
        if digest(release / name) != expected:
            raise ValueError('Immutable release artifact changed: ' + name)
    manifest_hash = digest(ROOT / cfg['split_manifest'])
    if approval.get('manifest_sha256') != manifest_hash:
        raise ValueError('Reviewed data manifest hash mismatch')
    review = approval.get('review_document')
    if not review or not (ROOT / review).is_file():
        raise ValueError('A durable leakage adjudication document is required')
    train, validation = rows_for(cfg, 'train'), rows_for(cfg, 'validation')
    if {r['image'] for r in train} & {r['image'] for r in validation}:
        raise ValueError('Training/validation image overlap')
    if {r['lesion_id'] for r in train} & {r['lesion_id'] for r in validation}:
        raise ValueError('Training/validation known lesion overlap')
    # Test metadata is checked for identity overlap only; its labels/pixels are
    # never supplied to the optimizer, loss, loader, weighting or teacher gate.
    with (ROOT / cfg['split_manifest']).open(newline='') as handle:
        all_rows = list(csv.DictReader(handle))
    seen_images, lesion_owner = set(), {}
    for row in all_rows:
        split = row['split']
        if split not in ['train', 'validation', 'test'] or row['image'] in seen_images or not row['lesion_id']:
            raise ValueError('Invalid or overlapping approved partition identities')
        seen_images.add(row['image'])
        if lesion_owner.setdefault(row['lesion_id'], split) != split:
            raise ValueError('Known lesion crosses an approved partition boundary')
    for split, values in [('train', train), ('validation', validation)]:
        if any(r['release_eligible'] != 'True' or r['original_split'] != split for r in values):
            raise ValueError('Quarantined or reassigned data supplied to trainer')
    frozen_weights = json.loads((release / 'class_weights.json').read_text())
    if {c: sum(r['label'] == c for r in train) for c in cfg['classes']} != frozen_weights['counts']:
        raise ValueError('Training class counts differ from release')
    return {'manifest_sha256': manifest_hash, 'review_sha256': digest(ROOT / review),
            'review_document': review}


def load_final_parent(path, expected_id, data_receipt):
    resolved = (ROOT / path).resolve()
    permitted = PROJECT / 'runs' / f'{expected_id}_seed42' / 'checkpoints' / 'best.pt'
    if resolved != permitted.resolve():
        raise ValueError('Only the declared fresh final parent is allowed')
    checkpoint = torch.load(resolved, map_location='cpu', weights_only=True)
    provenance = checkpoint['provenance']
    status = json.loads((permitted.parent.parent / 'run_status.json').read_text())
    if (provenance['project_id'] != PROJECT_ID or provenance['model_id'] != expected_id or
            provenance['artifact_kind'] != 'final_training_checkpoint' or
            status['status'] != 'complete' or status['run_id'] != provenance['run_id'] or
            provenance['data_receipt'] != data_receipt or
            not provenance.get('official_initialization_sha256')):
        raise ValueError('Fresh completed parent provenance/data checks failed')
    # Validate against the exact final configuration and completed-run receipt.
    if provenance['config_sha256'] != digest(PROJECT / 'configs' / f'{expected_id}.json'):
        raise ValueError('Final parent configuration changed')
    if status['best_checkpoint_sha256'] != digest(resolved):
        raise ValueError('Completed parent checkpoint hash mismatch')
    return checkpoint, {'path': path, 'sha256': digest(resolved), 'run_id': provenance['run_id'],
                        'validation_macro_f1': checkpoint['validation_macro_f1']}


def classification_metrics(matrix):
    matrix = matrix.float()
    support = matrix.sum(1)
    recall = matrix.diag() / support.clamp_min(1)
    return {'accuracy': float(matrix.diag().sum() / matrix.sum().clamp_min(1)),
            'macro_f1': macro_f1(matrix), 'balanced_accuracy': float(recall.mean()),
            'melanoma_recall': float(recall[1]), 'SCC_recall': float(recall[3])}


@torch.no_grad()
def validate(net, batches, prediction_path):
    net.eval()
    matrix = torch.zeros(4, 4, device='cuda', dtype=torch.long)
    losses, samples = 0.0, 0
    predictions = []
    for images, target, image_ids in batches:
        images, target = images.cuda(non_blocking=True), target.cuda(non_blocking=True)
        with autocast():
            logits = net(images)
        loss = torch.nn.functional.cross_entropy(logits.float(), target)
        matrix += torch.bincount(target * 4 + logits.argmax(-1), minlength=16).reshape(4, 4)
        losses += loss.item() * len(target)
        samples += len(target)
        probabilities = logits.float().softmax(-1).cpu().tolist()
        labels = target.cpu().tolist()
        for image_id, label, prob in zip(image_ids, labels, probabilities):
            predictions.append({'image': image_id, 'label_index': label,
                                **{name: prob[i] for i, name in enumerate(['NV', 'MEL', 'BCC', 'SCC'])}})
    with prediction_path.open('x', newline='') as handle:
        writer = csv.DictWriter(handle, ['image', 'label_index', 'NV', 'MEL', 'BCC', 'SCC'])
        writer.writeheader(); writer.writerows(predictions)
    return {**classification_metrics(matrix), 'unweighted_loss': losses / samples,
            'confusion_matrix': matrix.cpu().tolist(), 'samples': samples}


def update_summary(model_id, status):
    path = PROJECT / 'experiment_summary.json'
    summary = json.loads(path.read_text())
    summary['models'][model_id].update(status)
    temporary = path.with_suffix('.next')
    temporary.write_text(json.dumps(summary, indent=2) + '\n')
    os.replace(temporary, path)


def execute(cfg):
    data_receipt = approval_check(cfg)  # Before model loading or optimizer creation.
    authorization = json.loads((PROJECT / 'training_authorization.json').read_text())
    if authorization.get('authorized') is not True:
        raise RuntimeError('Phase5A is preparation only; subsequent explicit user authorization is required')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for final GPU runs')
    seed_all(cfg['seed'])
    torch.set_num_threads(4)
    checkpoint_dir = ROOT / cfg['checkpoint_dir']
    run_dir = checkpoint_dir.parent
    status_path = run_dir / 'run_status.json'
    if status_path.exists() or list(checkpoint_dir.glob('*.pt')):
        raise FileExistsError('Refusing to overwrite or resume any completed/partial final run')
    run_id = str(uuid.uuid4())
    provenance = {'project_id': PROJECT_ID, 'model_id': cfg['id'], 'run_id': run_id,
                  'artifact_kind': 'final_training_checkpoint', 'classes': cfg['classes'],
                  'config_sha256': digest(PROJECT / 'configs' / f'{cfg["id"]}.json'),
                  'data_receipt': data_receipt, 'parents': {}, 'official_initialization_sha256': None}
    teacher = None
    if cfg['id'] == 'M5':
        student_state, parent_student = load_final_parent(cfg['initialization']['student_checkpoint'], 'M3', data_receipt)
        teacher_state, parent_teacher = load_final_parent(cfg['initialization']['teacher_checkpoint'], 'M4', data_receipt)
        if not parent_teacher['validation_macro_f1'] > parent_student['validation_macro_f1'] + cfg['teacher_qualification']['min_macro_f1_improvement']:
            update_summary('M5', {'status': 'blocked_teacher_unsuitable', 'reason': 'Fresh M4 validation macro-F1 does not exceed fresh M3'})
            raise RuntimeError('M5 teacher qualification failed; do not assume a useful teacher')
        net = model(cfg)
        net.load_state_dict(student_state['state_dict'], strict=True)
        teacher = model(config('M4')).cuda().eval().requires_grad_(False)
        teacher.load_state_dict(teacher_state['state_dict'], strict=True)
        provenance['parents'] = {'student': parent_student, 'teacher': parent_teacher}
        provenance['official_initialization_sha256'] = student_state['provenance']['official_initialization_sha256']
        del student_state, teacher_state
    else:
        net = model(cfg, pretrained=True)
        provenance['official_initialization_sha256'] = json.loads((ROOT / 'final_training' / 'pretrained' / cfg['id'] / 'provenance.json').read_text())['sha256']
    net = net.cuda()
    train = loader(cfg, 'train', cfg['id'] == 'M5')
    validation = loader(cfg, 'validation', return_ids=True)
    counts, weights = train_weights(cfg)
    weights = weights.cuda()
    provenance['train_class_counts'] = counts
    provenance['train_class_weights'] = weights.cpu().tolist()
    (run_dir / 'resolved_config.json').write_text(json.dumps(cfg, indent=2) + '\n')
    scaler = torch.amp.GradScaler('cuda', enabled=precision() == torch.float16)
    best, stale, previous_stage = -1.0, 0, None
    global_updates = 0
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    status_path.write_text(json.dumps({'status': 'running', 'run_id': run_id, 'started_utc': datetime.now(timezone.utc).isoformat()}, indent=2))
    try:
        for epoch in range(cfg['max_total_epochs']):
            frozen = epoch < cfg['frozen_epochs']
            stage = 'head' if frozen else 'finetune'
            if stage != previous_stage:
                if previous_stage == 'head':
                    state = torch.load(checkpoint_dir / 'best.pt', map_location='cpu', weights_only=True)
                    net.load_state_dict(state['state_dict'], strict=True)
                    del state
                optimizer = configure_stage(net, cfg, frozen)
                stale = 0
                previous_stage = stage
            training_mode(net, cfg, frozen)
            total_loss, n, gate_total = 0.0, 0, 0.0
            train_matrix = torch.zeros(4, 4, device='cuda', dtype=torch.long)
            class_gate = torch.zeros(4, device='cuda', dtype=torch.long)
            for batch in train:
                batch = tuple(value.cuda(non_blocking=True) for value in batch)
                optimizer.zero_grad(set_to_none=True)
                with autocast():
                    loss, stats, logits = optimization_loss(net, batch, weights, cfg, teacher, return_logits=True)
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite training loss')
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                train_matrix += torch.bincount(batch[-1] * 4 + logits.detach().argmax(-1), minlength=16).reshape(4, 4)
                global_updates += 1
                size = len(batch[-1])
                total_loss += loss.item() * size
                n += size
                if stats:
                    gate_total += float(stats['gate_acceptance']) * size
                    class_gate += torch.bincount(batch[-1][stats['gate_mask']], minlength=4)
            metrics = validate(net, validation, run_dir / f'validation_predictions_epoch_{epoch+1:03d}.csv')
            record = {'epoch': epoch + 1, 'stage': stage, 'train_loss': total_loss / n,
                      'train': classification_metrics(train_matrix),
                      'peak_allocated_GiB': torch.cuda.max_memory_allocated() / 2**30,
                      'validation': metrics, 'global_updates': global_updates,
                      'gate_acceptance': gate_total / n if cfg['id'] == 'M5' else None,
                      'gate_acceptance_by_class': {name: float(class_gate[i]) / counts[i]
                                                   for i, name in enumerate(cfg['classes'])} if cfg['id'] == 'M5' else None,
                      'elapsed_seconds': time.perf_counter() - start}
            with (run_dir / 'history.jsonl').open('a') as handle:
                handle.write(json.dumps(record) + '\n')
            improved = metrics['macro_f1'] > best + cfg['early_stopping']['min_delta']
            if improved:
                best, stale = metrics['macro_f1'], 0
            else:
                stale += 1
            checkpoint = {'state_dict': {key: value.detach().cpu() for key, value in net.state_dict().items()},
                          'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(),
                          'provenance': provenance, 'epoch': epoch + 1, 'stage': stage,
                          'validation_macro_f1': metrics['macro_f1'], 'global_updates': global_updates}
            epoch_path = checkpoint_dir / f'epoch_{epoch+1:03d}.pt'
            with epoch_path.open('xb') as handle:
                torch.save(checkpoint, handle)
            for alias in (['last.pt', 'best.pt'] if improved else ['last.pt']):
                temporary = checkpoint_dir / (alias + '.next')
                temporary.symlink_to(epoch_path.name)
                os.replace(temporary, checkpoint_dir / alias)
            del checkpoint
            print(json.dumps(record), flush=True)
            if stale >= cfg['early_stopping']['patience'] and not frozen:
                break
        status_path.write_text(json.dumps({'status': 'complete', 'run_id': run_id,
                                           'epochs_completed': epoch + 1, 'best_validation_macro_f1': best,
                                           'elapsed_seconds': time.perf_counter() - start,
                                           'best_checkpoint_sha256': digest(checkpoint_dir / 'best.pt'),
                                           'global_updates': global_updates, 'budget_kind': cfg['budget_kind']}, indent=2) + '\n')
        update_summary(cfg['id'], json.loads(status_path.read_text()))
    except BaseException as exc:
        update_summary(cfg['id'], {'status': 'failed', 'run_id': run_id, 'error': repr(exc)})
        status_path.write_text(json.dumps({'status': 'failed', 'run_id': run_id, 'error': repr(exc)}, indent=2) + '\n')
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', choices=['M1', 'M2', 'M3', 'M4', 'M5'])
    parser.add_argument('--execute', action='store_true', help='Full run; requires reviewed dataset approval and a later explicit user instruction')
    args = parser.parse_args()
    cfg = config(args.model)
    if not args.execute:
        print(json.dumps(cfg, indent=2))
        print('INSPECTION ONLY: no optimization or checkpoint loading performed.')
        return
    execute(cfg)


if __name__ == '__main__':
    main()
