"""Validation-only FP32 training and provenance checks for the EDA extension.

Source hash globals can be replaced by literal hashes when embedding this source
in a notebook. Quick runs are isolated under ``run_dir/quick`` and may use a
shorter stopping budget. Official runs require the approved 100/10/64/.001
protocol and are saved under ``run_dir/official``.
"""
import copy
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from cuda_training import CapturedAdamStep
from eda_models import build_eda_decoder, count_parameters
from evaluate import compute_window_metrics
from provenance import array_hash, contract_hash, file_sha256, save_json
from sensing import generate_rademacher_matrix

# Explicit override points for self-contained notebook embedding.
EDA_MODEL_SOURCE_HASH = file_sha256(Path(__file__).with_name('eda_models.py'))
EDA_TRAINER_SOURCE_HASH = file_sha256(__file__)
LEGACY_MODEL_SOURCE_HASH = file_sha256(Path(__file__).with_name('model.py'))
LEGACY_TRAINER_SOURCE_HASH = file_sha256(Path(__file__).with_name('train.py'))
CUDA_STEP_SOURCE_HASH = file_sha256(Path(__file__).with_name('cuda_training.py'))


def eda_run_scope(config):
    return 'quick' if 'quick' in config.run_id.lower() else 'official'


def _validate_eda_training_config(config):
    scope = eda_run_scope(config)
    if config.batch_size != 64 or config.learning_rate != .001:
        raise ValueError('EDA training requires batch_size=64 and Adam learning_rate=.001')
    if scope == 'official' and (config.max_epochs, config.patience) != (100, 10):
        raise ValueError('The official stopping budget requires max_epochs=100, patience=10')
    if not 1 <= config.max_epochs <= 100 or not 1 <= config.patience <= 10:
        raise ValueError('Quick training budget must stay within the approved 100/10 limits')
    return scope


def _eda_set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def compute_eda_macro_prd(x, pred, subjects):
    """Window mean within each person, then an equally weighted person mean."""
    subjects = np.asarray(subjects)
    if subjects.ndim != 1 or len(subjects) != len(x) or len(subjects) == 0:
        raise ValueError('Validation subject labels must match nonempty windows')
    metrics = compute_window_metrics(x, pred)
    if metrics['decoder_failures'] or not np.isfinite(x).all():
        raise FloatingPointError('Nonfinite validation reference or prediction')
    means = []
    for subject in np.unique(subjects):
        mask = (subjects == subject) & metrics['valid_mask']
        if not mask.any():
            raise ValueError(f'Validation subject {subject} has no valid PRD windows')
        means.append(float(metrics['prd'][mask].mean()))
    return float(np.mean(means))


def _eda_validation_signature(model):
    # Training and load_state_dict update persistent tensors in place. Moving
    # or replacing tensors invalidates the graph, which must then be rebuilt.
    return tuple((tensor.data_ptr(), str(tensor.device), str(tensor.dtype), tuple(tensor.shape))
                 for tensor in list(model.parameters())+list(model.buffers()))


class CapturedValidationPredictor:
    """Persistent eval forward graph with unchanged FP32 cuDNN algorithms.

    It reads the model's current parameter/buffer storage on every replay;
    actual Adam updates therefore remain visible without another capture.
    Warmup runs in eval mode and cannot change BatchNorm running statistics.
    """

    def __init__(self, model, example_y):
        if example_y.device.type != 'cuda' or example_y.dtype != torch.float32:
            raise ValueError('Captured validation requires CUDA float32 inputs')
        model.eval()
        self.signature = _eda_validation_signature(model)
        self.y = example_y.clone()
        with torch.cuda.device(self.y.device), torch.inference_mode():
            current = torch.cuda.current_stream(self.y.device)
            stream = torch.cuda.Stream(device=self.y.device)
            stream.wait_stream(current)
            with torch.cuda.stream(stream):
                for _ in range(3):
                    model(self.y)
            current.wait_stream(stream)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.output = model(self.y)

    def matches(self, model, y):
        return (y.shape == self.y.shape and y.device == self.y.device and
                y.dtype == self.y.dtype and self.signature == _eda_validation_signature(model))

    def step(self, y):
        if y.shape != self.y.shape or y.device != self.y.device or y.dtype != self.y.dtype:
            raise ValueError('Captured validation batch shape/device/dtype mismatch')
        self.y.copy_(y)
        self.graph.replay()
        return self.output


def predict_eda_model(model, Y, device='cpu', batch_size=256):
    if len(Y) == 0:
        raise ValueError('Cannot predict empty measurements')
    model.eval()
    out = []
    with torch.inference_mode():
        for start in range(0, len(Y), batch_size):
            y = torch.as_tensor(Y[start:start+batch_size], dtype=torch.float32, device=device)
            # Keep the final shorter batch eager: no padding, duplicate windows
            # or changes to its convolution shape and resulting arithmetic.
            if y.device.type == 'cuda' and len(y) == batch_size:
                predictor = getattr(model, '_eda_validation_predictor', None)
                if predictor is None or not predictor.matches(model, y):
                    predictor = CapturedValidationPredictor(model, y)
                    model._eda_validation_predictor = predictor
                prediction = predictor.step(y)
            else:
                prediction = model(y)
            out.append(prediction.cpu().numpy())
    return np.concatenate(out)


def legacy_eda_checkpoint_contract(method, fold_data, M, seed, config, data_hash):
    """Exactly the original train.checkpoint_contract fields and source hashes."""
    if method not in ('cnn', 'linear'):
        raise ValueError('Only audited CNN and Linear can be loaded as legacy baselines')
    return {'method': method, 'M': M, 'seed': seed, 'fold': fold_data['fold'],
            'train_subjects': fold_data['train_subjects'], 'val_subjects': fold_data['val_subjects'],
            'test_subjects': fold_data['test_subjects'], 'mu_train': fold_data['mu_train'],
            'sigma_train': fold_data['sigma_train'], 'data_hash': data_hash,
            'numerical': config.numerical_contract(),
            'matrix_hash': array_hash(generate_rademacher_matrix(M)[1]),
            'model_source_hash': LEGACY_MODEL_SOURCE_HASH,
            'trainer_source_hash': LEGACY_TRAINER_SOURCE_HASH,
            'precision': 'fp32-no-tf32', 'cuda_step_source_hash': CUDA_STEP_SOURCE_HASH,
            'batch_size': config.batch_size, 'lr': config.learning_rate,
            'max_epochs': config.max_epochs, 'patience': config.patience}


def eda_checkpoint_contract(method, fold_data, M, seed, config, data_hash):
    if method not in ('cnn', 'linear', 'reslincnn_lite'):
        raise ValueError(f'Unknown EDA decoder {method}')
    scope = _validate_eda_training_config(config)
    # Begin from the original numerical fields so baseline and extension are
    # compared under the same preprocessing, matrix and training precision.
    base = legacy_eda_checkpoint_contract('cnn', fold_data, M, seed, config, data_hash)
    base.update(method=method, model_source_hash=EDA_MODEL_SOURCE_HASH,
                trainer_source_hash=EDA_TRAINER_SOURCE_HASH,
                baseline_model_source_hash=LEGACY_MODEL_SOURCE_HASH,
                scope=scope, run_id=config.run_id, loss='MSE', optimizer='Adam',
                train_windows_hash=array_hash(fold_data['train']['windows']),
                val_windows_hash=array_hash(fold_data['val']['windows']),
                val_subjects_hash=array_hash(fold_data['val']['subjects']))
    return base


def _validate_eda_history(ck, require_complete):
    history = ck.get('history', [])
    if not history or [row.get('epoch') for row in history] != list(range(1, len(history)+1)):
        raise ValueError('Checkpoint history must contain every actual epoch in order')
    for row in history:
        values = [row.get(k, float('nan')) for k in ('train_loss', 'val_macro_prd', 'epoch_seconds')]
        if not np.isfinite(values).all() or row['epoch_seconds'] < 0:
            raise ValueError('Checkpoint history contains invalid measurements')
    best_row = min(history, key=lambda row: row['val_macro_prd'])
    best_epoch = ck.get('best_epoch', ck.get('epoch'))
    if best_epoch != best_row['epoch'] or ck.get('val_macro_prd') != best_row['val_macro_prd']:
        raise ValueError('Checkpoint best epoch does not agree with validation history')
    if ck.get('epochs_run', len(history)) != len(history):
        raise ValueError('Checkpoint epochs_run does not agree with history')
    if not ck.get('model_state_dict') or not all(torch.isfinite(v).all() for v in ck['model_state_dict'].values()):
        raise ValueError('Checkpoint has missing or nonfinite model state')
    if require_complete or ck.get('complete'):
        if not ck.get('complete'):
            raise ValueError('Incomplete training checkpoint')
        contract = ck.get('contract', {})
        reason = ck.get('stop_reason')
        if reason == 'max_epochs':
            valid_stop = len(history) == contract.get('max_epochs')
        elif reason == 'patience':
            valid_stop = len(history)-best_epoch >= contract.get('patience', float('inf'))
        else:
            valid_stop = False
        if not valid_stop:
            raise ValueError('Complete checkpoint has an inconsistent stopping history')


def load_verified_eda_checkpoint(path, fingerprint, require_complete=True):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    if ck.get('fingerprint') != fingerprint or contract_hash(ck.get('contract')) != fingerprint:
        raise ValueError('Checkpoint fingerprint mismatch')
    _validate_eda_history(ck, require_complete)
    return ck


def load_legacy_eda_checkpoint(method, fd, M, seed, config, data_hash):
    """Read the original completed baseline, with no retraining or artifact writes."""
    contract = legacy_eda_checkpoint_contract(method, fd, M, seed, config, data_hash)
    path = Path(config.root)/'checkpoints'/'20261004_three_models_v2'/f'{method}_fold{fd["fold"]}_M{M}_seed{seed}.pt'
    # Same fingerprint/complete checks as original load_verified_checkpoint;
    # additionally require a coherent full history before reporting convergence.
    ck = load_verified_eda_checkpoint(path, contract_hash(contract))
    model = build_eda_decoder(method, M)
    model.load_state_dict(ck['model_state_dict'], strict=True)
    if ck.get('parameter_count') != count_parameters(model):
        raise ValueError('Legacy checkpoint parameter count mismatch')
    return path, ck


def _eda_atomic_torch_save(value, path):
    temporary = Path(str(path)+'.tmp')
    torch.save(value, temporary)
    os.replace(temporary, path)


def _eda_cpu_state(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _eda_cpu_state(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_eda_cpu_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_eda_cpu_state(item) for item in value)
    return copy.deepcopy(value)


def _validate_eda_inputs(fd, M, seed, config, Y_train, Y_val):
    if M not in config.M_list or seed not in config.seeds:
        raise ValueError('M and seed must belong to the locked experiment')
    groups = [set(fd[key]) for key in ('train_subjects', 'val_subjects', 'test_subjects')]
    if any(groups[i] & groups[j] for i, j in ((0, 1), (0, 2), (1, 2))):
        raise ValueError('Subject leakage across train/validation/test')
    for split, measurements in (('train', Y_train), ('val', Y_val)):
        targets = np.asarray(fd[split]['windows'])
        measurements = np.asarray(measurements)
        if targets.shape != (len(measurements), 256) or measurements.shape != (len(targets), M) or len(targets) == 0:
            raise ValueError('Training inputs require matching nonempty [windows,M] and [windows,256] arrays')
        if targets.dtype != np.float32 or measurements.dtype != np.float32:
            raise ValueError('EDA targets and dequantized measurements must be float32')
        if not np.isfinite(targets).all() or not np.isfinite(measurements).all():
            raise FloatingPointError('Nonfinite training input')
    if set(np.unique(fd['val']['subjects'])) != groups[1]:
        raise ValueError('Validation windows must cover every configured validation subject')


def train_eda_decoder(method, fold_data, M, seed, config, run_dir, Y_train, Y_val,
                      data_hash, device=None, verbose=True):
    """Train or resume one run, saving best validation state and all real epochs."""
    scope = _validate_eda_training_config(config)
    _validate_eda_inputs(fold_data, M, seed, config, Y_train, Y_val)
    device = str(torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu')))
    is_cuda = torch.device(device).type == 'cuda'
    contract = eda_checkpoint_contract(method, fold_data, M, seed, config, data_hash)
    # Fingerprint actual shared measurement inputs, preventing accidental reuse
    # of a run trained from unquantized or otherwise different measurements.
    contract.update(train_measurements_hash=array_hash(Y_train), val_measurements_hash=array_hash(Y_val))
    fingerprint = contract_hash(contract)
    directory = Path(run_dir)/scope
    directory.mkdir(parents=True, exist_ok=True)
    name = f'{method}_fold{fold_data["fold"]}_M{M}_seed{seed}'
    path, resume_path = directory/(name+'.pt'), directory/(name+'.resume.pt')
    history_path = directory/(name+'.history.json')
    if path.exists():
        existing = load_verified_eda_checkpoint(path, fingerprint, require_complete=False)
        if existing['complete']:
            if verbose:
                print(f'REUSE {name} epochs_run={existing["epochs_run"]} best_epoch={existing["best_epoch"]}', flush=True)
            return path
        if not resume_path.exists():
            raise ValueError('Incomplete checkpoint has no recoverable optimizer/RNG state')
    _eda_set_seed(seed)
    start = time.perf_counter()
    model = build_eda_decoder(method, M).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, capturable=is_cuda)
    x = torch.as_tensor(fold_data['train']['windows'], dtype=torch.float32, device=device)
    y = torch.as_tensor(Y_train, dtype=torch.float32, device=device)
    best, best_epoch, stale = float('inf'), 0, 0
    best_state, best_optimizer, history = None, None, []
    elapsed_prior = 0.
    if resume_path.exists():
        resume = load_verified_eda_checkpoint(resume_path, fingerprint, require_complete=False)
        if resume['device'] != device:
            raise ValueError('Resuming training requires the original training device')
        model.load_state_dict(resume['current_state'], strict=True)
        optimizer.load_state_dict(resume['optimizer_state'])
        best, best_epoch, stale = resume['val_macro_prd'], resume['best_epoch'], resume['stale']
        best_state, best_optimizer = resume['model_state_dict'], resume['optimizer_state_dict']
        history, elapsed_prior = resume['history'], resume['training_seconds']
        torch.set_rng_state(resume['torch_rng'])
        np.random.set_state(resume['numpy_rng'])
        random.setstate(resume['python_rng'])
        if is_cuda:
            torch.cuda.set_rng_state_all(resume['cuda_rng'])
    if is_cuda:
        torch.cuda.reset_peak_memory_stats(device)
    captured = CapturedAdamStep(model, optimizer, y[:config.batch_size], x[:config.batch_size]) if is_cuda else None
    if verbose:
        print(f'TRAIN {name} scope={scope} windows={len(x)} validation={len(Y_val)} device={device}', flush=True)
    state = None
    for epoch in range(len(history)+1, config.max_epochs+1):
        if stale >= config.patience:
            break
        epoch_start = time.perf_counter()
        model.train()
        order = torch.randperm(len(x), device=device)
        loss_sum = torch.zeros((), device=device)
        for indices in order.split(config.batch_size):
            if captured is not None:
                loss = captured.step(y[indices], x[indices])
            else:
                optimizer.zero_grad(set_to_none=True)
                loss = (model(y[indices])-x[indices]).square().mean()
                loss.backward()
                optimizer.step()
            loss_sum += loss.detach()*len(indices)
        if not torch.isfinite(loss_sum):
            raise FloatingPointError(f'Nonfinite training loss in epoch {epoch}')
        prediction = predict_eda_model(model, Y_val, device)
        score = compute_eda_macro_prd(fold_data['val']['windows'], prediction, fold_data['val']['subjects'])
        row = {'epoch': epoch, 'train_loss': float(loss_sum.cpu())/len(x),
               'val_macro_prd': score, 'epoch_seconds': time.perf_counter()-epoch_start}
        history.append(row)
        if score < best:
            best, best_epoch, stale = score, epoch, 0
            best_state = _eda_cpu_state(model.state_dict())
            best_optimizer = _eda_cpu_state(optimizer.state_dict())
        else:
            stale += 1
        state = {'complete': False, 'fingerprint': fingerprint, 'contract': contract,
                 'model_state_dict': best_state, 'optimizer_state_dict': best_optimizer,
                 'model_name': method, 'fold': fold_data['fold'], 'M': M, 'seed': seed,
                 'val_macro_prd': best, 'epoch': best_epoch, 'best_epoch': best_epoch,
                 'epochs_run': len(history), 'history': history, 'stale': stale,
                 'training_seconds': elapsed_prior+time.perf_counter()-start,
                 'mean_epoch_seconds': float(np.mean([h['epoch_seconds'] for h in history])),
                 'parameter_count': count_parameters(model), 'device': device,
                 'gpu': torch.cuda.get_device_name(device) if is_cuda else None,
                 'peak_gpu_bytes': torch.cuda.max_memory_allocated(device) if is_cuda else 0}
        resume = dict(state, current_state=_eda_cpu_state(model.state_dict()),
                      optimizer_state=_eda_cpu_state(optimizer.state_dict()), torch_rng=torch.get_rng_state(),
                      numpy_rng=np.random.get_state(), python_rng=random.getstate(),
                      cuda_rng=torch.cuda.get_rng_state_all() if is_cuda else [])
        _eda_atomic_torch_save(resume, resume_path)
        _eda_atomic_torch_save(state, path)
        save_json(history_path, {k: v for k, v in state.items() if k not in ('model_state_dict', 'optimizer_state_dict')})
        if verbose:
            print(f'{name} epoch={epoch} train_mse={row["train_loss"]:.7g} '
                  f'val_macro_prd={score:.6f}% best_epoch={best_epoch} stale={stale}', flush=True)
    if state is None:
        state = load_verified_eda_checkpoint(resume_path, fingerprint, require_complete=False)
        state = {k: v for k, v in state.items() if k not in ('current_state', 'optimizer_state', 'torch_rng', 'numpy_rng', 'python_rng', 'cuda_rng')}
    state.update(complete=True, stop_reason='patience' if stale >= config.patience else 'max_epochs',
                 training_seconds=elapsed_prior+time.perf_counter()-start)
    _validate_eda_history(state, True)
    _eda_atomic_torch_save(state, path)
    metadata = {k: v for k, v in state.items() if k not in ('model_state_dict', 'optimizer_state_dict')}
    metadata['checkpoint_sha256'] = file_sha256(path)
    save_json(history_path, metadata)
    if resume_path.exists():
        resume_path.unlink()
    if verbose:
        print(f'COMPLETE {name} epochs_run={len(history)} best_epoch={best_epoch} '
              f'best_val_macro_prd={best:.6f}% total_seconds={state["training_seconds"]:.3f}', flush=True)
    return path
