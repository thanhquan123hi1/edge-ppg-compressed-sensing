"""PPG-DaLiA data, versioned Float32 causal cache and subject-wise folds."""
import json
import pickle
from pathlib import Path
import numpy as np
import scipy.signal
from experiment_config import ExperimentConfig,filter_sos
from provenance import file_sha256,contract_hash,save_json

SUBJECTS = [f'S{i}' for i in range(1,16)]
FOLD_CONFIGS = {
    1:{'train':[f'S{i}' for i in range(7,16)],'val':['S4','S5','S6'],'test':['S1','S2','S3']},
    2:{'train':['S1','S2','S3']+[f'S{i}' for i in range(10,16)],'val':['S7','S8','S9'],'test':['S4','S5','S6']},
    3:{'train':[f'S{i}' for i in range(1,7)]+['S13','S14','S15'],'val':['S10','S11','S12'],'test':['S7','S8','S9']},
    4:{'train':[f'S{i}' for i in range(1,10)],'val':['S13','S14','S15'],'test':['S10','S11','S12']},
    5:{'train':[f'S{i}' for i in range(4,13)],'val':['S1','S2','S3'],'test':['S13','S14','S15']}}

def load_raw_bvp(subject,data_root):
    path = Path(data_root)/subject/f'{subject}.pkl'
    with path.open('rb') as f:
        d = pickle.load(f,encoding='latin1')
    return np.asarray(d['signal']['wrist']['BVP'],dtype=np.float32).ravel()

def filter_record(raw,sos=None,drop=320):
    """Reset at each finite segment; retain positions and mark gaps/transients NaN."""
    raw = np.asarray(raw,dtype=np.float32)
    sos = filter_sos() if sos is None else np.asarray(sos,dtype=np.float32)
    edges = np.diff(np.r_[False,np.isfinite(raw),False].astype(np.int8))
    starts,stops = np.where(edges == 1)[0],np.where(edges == -1)[0]
    out = np.full(raw.shape,np.nan,dtype=np.float32)
    for start,stop in zip(starts,stops):
        if stop-start > drop:
            y = scipy.signal.sosfilt(sos,raw[start:stop],zi=np.zeros((2,2),dtype=np.float32))[0]
            out[start+drop:stop] = y[drop:]
    return out[drop:]

def cache_matches(path,expected):
    meta = Path(str(path)+'.json')
    if not Path(path).exists() or not meta.exists():
        return False
    try:
        saved = json.loads(meta.read_text(encoding='utf-8'))
        return saved.get('fingerprint') == contract_hash(expected) and saved.get('cache_sha256') == file_sha256(path)
    except (ValueError,OSError):
        return False

def load_or_build_dataset(config):
    cache = Path(config.root)/'data'/'filtered_bvp_v2.npz'
    raw_hashes = {s:file_sha256(Path(config.data_root)/s/f'{s}.pkl') for s in SUBJECTS}
    contract = {'numerical':config.numerical_contract(),'raw_sha256':raw_hashes}
    if cache_matches(cache,contract):
        with np.load(cache) as z:
            return {s:z[s] for s in SUBJECTS},json.loads(Path(str(cache)+'.json').read_text(encoding='utf-8'))
    data={}; records=[]
    for s in SUBJECTS:
        raw = load_raw_bvp(s,config.data_root)
        data[s] = filter_record(raw,drop=config.drop)
        records.append({'subject':s,'raw_samples':len(raw),'cached_samples':len(data[s]),
                        'nonfinite_raw':int(np.sum(~np.isfinite(raw))),
                        'nonfinite_filtered':int(np.sum(~np.isfinite(data[s])))})
        print(f'DATA {s}: raw={len(raw)}, cached={len(data[s])}',flush=True)
    cache.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(cache,**data)
    meta={'schema':2,'fingerprint':contract_hash(contract),'contract':contract,'records':records,'cache_sha256':file_sha256(cache)}
    save_json(str(cache)+'.json',meta)
    return data,meta

def load_filtered_dataset(cache_path='D:/IOT/CuoiKy/data/filtered_bvp_v2.npz'):
    with np.load(cache_path) as z:
        return {k:z[k] for k in z.files}

def extract_windows_for_signal(signal,N=256,H=128):
    signal=np.ascontiguousarray(signal,dtype=np.float32)
    if len(signal)<N:
        return np.empty((0,N),dtype=np.float32)
    return np.ascontiguousarray(np.lib.stride_tricks.sliding_window_view(signal,N)[::H])

def prepare_fold_data(fold,cache_path=None,N=256,H=128,dataset=None):
    if fold not in FOLD_CONFIGS:
        raise ValueError('Invalid fold')
    data = dataset if dataset is not None else load_filtered_dataset() if cache_path is None else load_filtered_dataset(cache_path)
    cfg=FOLD_CONFIGS[fold]
    train=np.concatenate([data[s][np.isfinite(data[s])] for s in cfg['train']])
    mu=float(np.mean(train,dtype=np.float64)); sigma=float(np.std(train,dtype=np.float64))
    if not np.isfinite(sigma) or sigma<=1e-8:
        raise ValueError('Invalid train sigma')
    result={'fold':fold,'mu_train':mu,'sigma_train':sigma,
            **{f'{k}_subjects':cfg[k] for k in ['train','val','test']}}
    for split in ['train','val','test']:
        windows=[]; subjects=[]; indices=[]; invalid=0;invalid_records=[]
        for s in cfg[split]:
            z=((data[s]-np.float32(mu))/np.float32(sigma)).astype(np.float32)
            w=extract_windows_for_signal(z,N,H)
            valid=np.isfinite(w).all(axis=1); idx=np.flatnonzero(valid)
            invalid_records.extend({'subject':s,'window_index':int(i),'raw_start_sample':320+H*int(i),'failure_reason':'nonfinite_reference'} for i in np.flatnonzero(~valid))
            invalid+=int(np.sum(~valid)); windows.append(w[valid]);subjects.extend([s]*len(idx));indices.extend(idx.tolist())
        result[split]={'windows':np.vstack(windows),'subjects':np.asarray(subjects),
                       'window_indices':np.asarray(indices,dtype=np.int32),
                       'raw_start_samples':320+H*np.asarray(indices,dtype=np.int64),
                       'invalid_reference_windows':invalid,'invalid_reference_records':invalid_records}
    return result
