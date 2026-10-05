"""Canonical v2 validation replay; no legacy cache or normalization adapter.

Float32 matmul can differ across BLAS/GPU implementations. Measurement hashes
are reported verbatim, never rewritten. The 0.01 percentage-point replay gate
is fixed before inspecting test predictions, not fitted to their performance.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from evaluate import compute_window_metrics, aggregate_subject_metrics
from notebook_lite_extension import predict_lite
from provenance import array_hash, file_sha256
from sensing import generate_rademacher_matrix, project_measurements
from quantization import quantize_int16


def require_complete_inventory(inventory, models):
    if len(inventory) != 45 or len(models) != 45 or not inventory.status.eq('VERIFIED').all():
        raise ValueError('Main comparison requires all 45 completed, contract-checked Lite checkpoints')


def replay_canonical_validation(models, folds, directory, device='cpu', tolerance=0.01):
    """Reproduce saved best-state validation on the canonical fold, person first."""
    rows = []
    destination = Path(directory) / 'canonical_lite_validation_audit.csv'
    for fold, M in sorted({(f, m) for f, m, _ in models}):
        fd = folds[fold]
        x = fd['val']['windows']
        _, phi = generate_rademacher_matrix(M)
        _, _, y, clips = quantize_int16(project_measurements(x, phi))
        local_hash = array_hash(y)
        for key, record in sorted(models.items()):
            if key[:2] != (fold, M):
                continue
            ck = record['checkpoint']
            prediction = predict_lite(record['model'], y, device=device, batch_size=512)
            metrics = compute_window_metrics(x, prediction)
            subjects = aggregate_subject_metrics(metrics, fd['val']['subjects'])
            measured = float(np.mean([v['prd'] for v in subjects.values()]))
            difference = abs(measured - ck['val_macro_prd'])
            passed = np.isfinite(difference) and difference <= tolerance and not metrics['decoder_failures']
            rows.append({'Fold': fold, 'M': M, 'Seed': key[2], 'path': str(record['path']),
                         'checkpoint_sha256': file_sha256(record['path']), 'fingerprint': ck['fingerprint'],
                         'published_validation_prd': ck['val_macro_prd'], 'replayed_validation_prd': measured,
                         'absolute_difference_percentage_points': difference,
                         'tolerance_percentage_points': tolerance, 'status': 'PASS' if passed else 'FAIL',
                         'training_complete_verified': bool(ck['complete']), 'epochs_run': ck['epochs_run'],
                         'best_epoch': ck['best_epoch'], 'stop_reason': ck['stop_reason'],
                         'local_val_measurements_hash': local_hash,
                         'published_val_measurements_hash': ck.get('contract', {}).get('val_measurements_hash'),
                         'measurement_hashes_bit_identical': local_hash == ck.get('contract', {}).get('val_measurements_hash'),
                         'validation_clip_count': clips, 'decoder_failures': metrics['decoder_failures']})
            pd.DataFrame(rows).to_csv(destination, index=False)
            if not passed:
                raise ValueError(f'Canonical validation replay disagrees: {record["path"]}: {difference:.9g} pp')
            print(f'CANONICAL_LITE fold={fold} M={M} seed={key[2]}: validation={measured:.6f}% delta={difference:.9g} pp', flush=True)
    return pd.DataFrame(rows)


def export_training_evidence(models, directory):
    """Export actual histories and hardware metadata; compare no unlike GPUs."""
    budgets, histories = [], []
    for (fold, M, seed), record in sorted(models.items()):
        ck = record['checkpoint']
        budgets.append({'Fold': fold, 'M': M, 'Seed': seed, 'best_epoch': ck['best_epoch'],
                        'epochs_run': ck['epochs_run'], 'stop_reason': ck['stop_reason'],
                        'training_seconds': ck['training_seconds'], 'device': ck.get('device'),
                        'gpu': ck.get('gpu'), 'parameter_count': ck['parameter_count'],
                        'published_peak_gpu_bytes': ck.get('peak_gpu_bytes'), 'fingerprint': ck['fingerprint']})
        histories.extend({'Fold': fold, 'M': M, 'Seed': seed, **row} for row in ck['history'])
    table = pd.DataFrame(budgets)
    table.to_csv(Path(directory) / 'lite_training_completion.csv', index=False)
    pd.DataFrame(histories).to_csv(Path(directory) / 'lite_full_training_history.csv', index=False)
    return table
