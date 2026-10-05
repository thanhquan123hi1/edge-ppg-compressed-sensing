"""Verified Colab checkpoint ingestion and evaluation for the existing notebook.

No training is triggered here. Missing/rejected runs never become macro scores.
"""
from dataclasses import replace
from pathlib import Path
import json
import pickle
import time
import numpy as np
import pandas as pd
import torch
from dataset import FOLD_CONFIGS, SUBJECTS
from eda_models import ResLinCNNLite, count_parameters
from eda_training import eda_checkpoint_contract, _validate_eda_history
from evaluate import compute_window_metrics, aggregate_subject_metrics, compute_macro_statistics
from provenance import array_hash, contract_hash, file_sha256, save_json
from sensing import generate_rademacher_matrix, project_measurements
from quantization import quantize_int16

MS=(51,77,128)
SEEDS=(11,22,33)

def json_records(frame):
    """Strict JSON: missing scores are null; valid infinite SNR stays explicit."""
    rows=[]
    for row in frame.to_dict('records'):
        rows.append({k:(None if np.isnan(v) else '+inf' if v>0 else '-inf')
                     if isinstance(v,(float,np.floating)) and not np.isfinite(v) else v for k,v in row.items()})
    return rows

def expected_lite_contract(fd,M,seed,config,data_hash,run_id):
    cfg=replace(config,run_id=run_id)
    contract=eda_checkpoint_contract('reslincnn_lite',fd,M,seed,cfg,data_hash)
    _,phi=generate_rademacher_matrix(M,config.N,config.matrix_seed)
    for split in ('train','val'):
        y=quantize_int16(project_measurements(fd[split]['windows'],phi))[2]
        contract[f'{split}_measurements_hash']=array_hash(y)
    return contract

def load_lite_checkpoint(path,fd,M,seed,config,data_hash):
    ck=torch.load(path,map_location='cpu',weights_only=True)
    if not isinstance(ck,dict) or not isinstance(ck.get('contract'),dict) or not ck.get('fingerprint'):
        raise ValueError('Missing checkpoint provenance/contract; weights alone are insufficient')
    contract=ck['contract'];run_id=contract.get('run_id')
    for key in ('train_measurements_hash','val_measurements_hash'):
        digest=contract.get(key)
        if not isinstance(digest,str) or len(digest)!=64 or any(c not in '0123456789abcdefABCDEF' for c in digest):
            raise ValueError('Missing or malformed measurement hash; cannot attribute it to BLAS')
    if not isinstance(run_id,str) or not run_id or 'quick' in run_id.lower() or contract.get('scope')!='official':
        raise ValueError('Only a completed official run is eligible')
    expected=expected_lite_contract(fd,M,seed,config,data_hash,run_id)
    if contract_hash(contract)!=ck['fingerprint']:
        raise ValueError('Checkpoint contract differs from data/split/normalization/quantizer/architecture/training protocol')
    if contract!=expected:
        measurement_keys={'train_measurements_hash','val_measurements_hash'}
        differing={k for k in set(contract)|set(expected) if contract.get(k)!=expected.get(k)}
        if not differing.issubset(measurement_keys):
            raise ValueError('Checkpoint contract differs from data/split/normalization/quantizer/architecture/training protocol')
        ck=dict(ck,measurement_hash_cross_platform_blas=True)
    _validate_eda_history(ck,True)
    scores=np.array([row['val_macro_prd'] for row in ck['history']],dtype=float)
    losses=np.array([row['train_loss'] for row in ck['history']],dtype=float)
    if (scores<0).any() or (losses<0).any() or len(scores)>config.max_epochs:
        raise ValueError('Invalid checkpoint history/budget')
    if ck['best_epoch']!=int(np.argmin(scores))+1:
        raise ValueError('Checkpoint must select the first minimum validation epoch')
    if ck['stop_reason']=='patience' and len(scores)-ck['best_epoch']!=config.patience:
        raise ValueError('Inconsistent full early-stopping history')
    running_best=float('inf');stale=0
    for epoch,score in enumerate(scores,1):
        if score<running_best:running_best=score;stale=0
        else:stale+=1
        if stale>=config.patience and (epoch!=len(scores) or ck['stop_reason']!='patience'):
            raise ValueError('History must end at the first patience stopping trigger')
    if (ck.get('fold'),ck.get('M'),ck.get('seed'),ck.get('model_name'))!=(fd['fold'],M,seed,'reslincnn_lite'):
        raise ValueError('Checkpoint identity mismatch')
    if not np.isfinite(ck.get('training_seconds',np.nan)) or ck['training_seconds']<=0:
        raise ValueError('Missing measured training duration')
    model=ResLinCNNLite(M)
    state=ck['model_state_dict']
    if any(v.dtype!=torch.float32 or not torch.isfinite(v).all() for v in state.values()):
        raise ValueError('Nonfinite/non-Float32 checkpoint state')
    model.load_state_dict(state,strict=True)
    if ck.get('parameter_count')!=count_parameters(model):
        raise ValueError('Checkpoint parameter count mismatch')
    return model.eval(),ck

def collect_lite_checkpoints(directory,folds,config,data_hash):
    directory=Path(directory);rows=[];models={}
    for fold in range(1,6):
        for M in MS:
            for seed in SEEDS:
                path=directory/f'reslincnn_lite_fold{fold}_M{M}_seed{seed}.pt'
                row={'Fold':fold,'M':M,'Seed':seed,'status':'MISSING','path':str(path),'detail':'Chưa có trọng số Colab'}
                if path.is_file():
                    try:
                        model,ck=load_lite_checkpoint(path,folds[fold],M,seed,config,data_hash)
                        models[(fold,M,seed)]={'model':model,'checkpoint':ck,'path':path}
                        row.update(status='VERIFIED',detail='Đủ provenance và lịch sử train',
                                   checkpoint_sha256=file_sha256(path),best_epoch=ck['best_epoch'],
                                   epochs_run=ck['epochs_run'],training_seconds=ck['training_seconds'])
                    except (ValueError,RuntimeError,KeyError,TypeError,EOFError,pickle.UnpicklingError,OSError) as exc:
                        row.update(status='REJECTED',detail=str(exc))
                rows.append(row)
    return pd.DataFrame(rows),models

def predict_lite(model,y,device='cpu',batch_size=256):
    if len(y)==0 or y.ndim!=2 or not np.isfinite(y).all():
        raise ValueError('Expected nonempty finite measurement matrix')
    model=model.to(device).eval();pred=[]
    with torch.inference_mode():
        for start in range(0,len(y),batch_size):
            pred.append(model(torch.from_numpy(y[start:start+batch_size]).to(device)).cpu().numpy())
    return np.concatenate(pred)

def evaluate_lite_checkpoints(models,folds,directory,data_hash,device='cpu'):
    out=Path(directory)/'lite_windows';out.mkdir(parents=True,exist_ok=True)
    results={}
    for (fold,M,seed),record in models.items():
        fd=folds[fold];x=fd['test']['windows'];_,phi=generate_rademacher_matrix(M)
        _,_,y,clips=quantize_int16(project_measurements(x,phi))
        pred=predict_lite(record['model'],y,device)
        metrics=compute_window_metrics(x,pred)
        path=out/f'reslincnn_lite_fold{fold}_M{M}_seed{seed}.npz'
        np.savez_compressed(path,predictions=pred,prd=metrics['prd'],rmse=metrics['rmse'],
                            snr_rec=metrics['snr_rec'],subject=fd['test']['subjects'],
                            raw_start_sample=fd['test']['raw_start_samples'])
        save_json(str(path)+'.json',{'data_sha256':data_hash,'checkpoint_sha256':file_sha256(record['path']),
                  'result_sha256':file_sha256(path),'fingerprint':record['checkpoint']['fingerprint'],
                  'quantized_values':y.size,'clip_count':clips,'decoder_failures':metrics['decoder_failures']})
        # Save the failing case, then refuse to manufacture a successful score.
        sub=aggregate_subject_metrics(metrics,fd['test']['subjects'])
        results[(fold,M,seed)]={'predictions':pred,'window_metrics':metrics,'subject_metrics':sub,
                              'checkpoint_sha256':file_sha256(record['path']),'clip_count':clips,'quantized_values':y.size}
    return results

def summarize_lite_results(results):
    rows=[];subjects={};seed_rows=[]
    for M in MS:
        keys=[(f,M,s) for f in range(1,6) for s in SEEDS]
        completed=[k for k in keys if k in results]
        row={'M':M,'status':'COMPLETE' if len(completed)==15 else 'INCOMPLETE' if completed else 'NOT_RUN',
             'runs_complete':len(completed),'runs_expected':15,'subjects_complete':0,
             'macro_prd':np.nan,'subject_std':np.nan,'rmse':np.nan,'snr_rec':np.nan}
        for f,_,seed in completed:
            if set(results[(f,M,seed)]['subject_metrics'])!=set(FOLD_CONFIGS[f]['test']):
                raise ValueError('Missing or unexpected test subject/person in a Lite run')
        if len(completed)==15:
            per_person={}
            for f in range(1,6):
                for subject in FOLD_CONFIGS[f]['test']:
                    per_person[subject]={metric:float(np.mean([results[(f,M,s)]['subject_metrics'][subject][metric] for s in SEEDS]))
                                         for metric in ('prd','rmse','snr_rec')}
            if set(per_person)!=set(SUBJECTS):raise ValueError('Expected all 15 people')
            stat=compute_macro_statistics(per_person)
            row.update(subjects_complete=15,macro_prd=stat['macro_prd_mean'],subject_std=stat['macro_prd_std'],
                       rmse=stat['macro_rmse_mean'],snr_rec=stat['macro_snr_mean'])
            subjects[M]=per_person
            for seed in SEEDS:
                seed_sub={s:metrics for f in range(1,6) for s,metrics in results[(f,M,seed)]['subject_metrics'].items()}
                stat=compute_macro_statistics(seed_sub)
                seed_rows.append({'M':M,'Seed':seed,'macro_prd':stat['macro_prd_mean'],'subject_std':stat['macro_prd_std']})
        rows.append(row)
    return pd.DataFrame(rows),subjects,pd.DataFrame(seed_rows,columns=['M','Seed','macro_prd','subject_std'])

def table_with_lite(baseline,summary,cpu=None):
    table=baseline.copy();table['Trạng thái']='ĐỦ KẾT QUẢ';table['Lượt đánh giá']=table.Decoder.map(lambda d:'5/5' if 'OMP' in d else '15/15')
    def fmt(value,digits):return f'{value:.{digits}f}' if not np.isnan(value) else 'CHƯA ĐỦ KẾT QUẢ'
    for row in summary.to_dict('records'):
        M=row['M'];lat=(cpu or {}).get(M,{}).get('ResLinCNN_Lite')
        table=pd.concat([table,pd.DataFrame([{'M':M,'Decoder':'ResLinCNN_Lite','Macro-PRD (%)':fmt(row['macro_prd'],2),
            'Subject Std (%)':fmt(row['subject_std'],2),'RMSE':fmt(row['rmse'],4),'SNRrec (dB)':fmt(row['snr_rec'],2),
            'CPU p50 (ms)':f'{lat["p50_ms"]:.3f}' if lat else 'CHƯA ĐO',
            'CPU p95 (ms)':f'{lat["p95_ms"]:.3f}' if lat else 'CHƯA ĐO',
            'Trạng thái':row['status'],'Lượt đánh giá':f'{row["runs_complete"]}/15'}])],ignore_index=True)
    return table.sort_values('M',kind='stable').reset_index(drop=True)

def benchmark_three_decoders(folds,omp_results,lite_models,checkpoint_directory,read_checkpoint,
                             directory,warmup=100,evaluations=1000,sessions=3):
    from baseline_omp import OMPDecoder
    from model import CNN1DDecoder
    from threadpoolctl import threadpool_limits,threadpool_info
    torch.set_num_threads(1)
    timings=[];latencies={}
    with threadpool_limits(limits=1):
        print('CPU benchmark batch=1; torch_threads=',torch.get_num_threads(),'BLAS=',threadpool_info())
        for M in MS:
            _,phi=generate_rademacher_matrix(M)
            _,_,y,_=quantize_int16(project_measurements(folds[1]['test']['windows'],phi))
            if len(y)<evaluations:raise ValueError('Not enough distinct benchmark inputs')
            indices=np.linspace(0,len(y)-1,evaluations,dtype=int);inputs=y[indices]
            dec=OMPDecoder(phi);k=omp_results[(1,M)]['selected_k']
            cnn=CNN1DDecoder(M).eval()
            ck=read_checkpoint(Path(checkpoint_directory)/f'cnn_fold1_M{M}_seed11.pt',folds[1],M,11,device='cpu')
            cnn.load_state_dict(ck['model_state_dict'],strict=True)
            methods={'omp':lambda v:dec.decode_single(v,K_max=k),
                     'cnn':lambda v:cnn(torch.from_numpy(v[None,:]))}
            record=lite_models.get((1,M,11))
            if record:
                lite=record['model'].cpu().eval()
                methods['reslincnn_lite']=lambda v:lite(torch.from_numpy(v[None,:]))
            else:print(f'M={M}: ResLinCNN_Lite CHƯA ĐO (thiếu checkpoint Fold1/seed11 đã xác minh)')
            names=list(methods)
            with torch.inference_mode():
                for session in range(sessions):
                    order=names[session%len(names):]+names[:session%len(names)]
                    for name in order:
                        call=methods[name]
                        for _ in range(warmup):call(inputs[0])
                        for index,value in zip(indices,inputs):
                            start=time.perf_counter();call(value);ms=(time.perf_counter()-start)*1000
                            timings.append({'M':M,'method':name,'session':session,'input_index':int(index),'ms':ms})
            latencies[M]={}
            for name,label in [('omp','OMP'),('cnn','CNN'),('reslincnn_lite','ResLinCNN_Lite')]:
                ms=np.asarray([t['ms'] for t in timings if t['M']==M and t['method']==name])
                if len(ms):latencies[M][label]={'mean_ms':float(ms.mean()),'p50_ms':float(np.percentile(ms,50)),'p95_ms':float(np.percentile(ms,95))}
    frame=pd.DataFrame(timings);frame.to_csv(Path(directory)/'cpu_decoder_samples.csv',index=False)
    save_json(Path(directory)/'cpu_benchmark_protocol.json',{'batch_size':1,'threads':1,'warmup':warmup,
              'samples_per_method_per_session':evaluations,'sessions':sessions,'method_order':'rotated by session',
              'device':'cpu','same_input_indices':True})
    return latencies
