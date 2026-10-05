"""Locked FP32 training; validation-only selection; verified resumable artifacts."""
import copy
import os
import random
import time
from pathlib import Path
import numpy as np
import torch
from model import build_decoder, count_parameters
from evaluate import compute_window_metrics
from provenance import contract_hash, array_hash, file_sha256, save_json
from cuda_training import CapturedAdamStep

def set_seed(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False

def compute_macro_prd(x,pred,subjects):
    metrics=compute_window_metrics(x,pred)
    if metrics['decoder_failures']:raise FloatingPointError('Nonfinite validation prediction')
    means=[np.mean(metrics['prd'][(subjects==s)&metrics['valid_mask']]) for s in np.unique(subjects)]
    return float(np.mean(means))

def checkpoint_contract(method,fold_data,M,seed,config,data_hash):
    return {'method':method,'M':M,'seed':seed,'fold':fold_data['fold'],
        'train_subjects':fold_data['train_subjects'],'val_subjects':fold_data['val_subjects'],
        'test_subjects':fold_data['test_subjects'],'mu_train':fold_data['mu_train'],
        'sigma_train':fold_data['sigma_train'],'data_hash':data_hash,
        'numerical':config.numerical_contract(),'matrix_hash':array_hash(__import__('sensing').generate_rademacher_matrix(M)[1]),
        'model_source_hash':file_sha256(Path(__file__).with_name('model.py')),
        'trainer_source_hash':file_sha256(__file__),'precision':'fp32-no-tf32',
        'cuda_step_source_hash':file_sha256(Path(__file__).with_name('cuda_training.py')),
        'batch_size':config.batch_size,'lr':config.learning_rate,'max_epochs':config.max_epochs,'patience':config.patience}

def load_verified_checkpoint(path,fingerprint,require_complete=True):
    ck=torch.load(path,map_location='cpu',weights_only=False)
    if ck.get('fingerprint')!=fingerprint:raise ValueError(f'Checkpoint fingerprint mismatch: {path}')
    if require_complete and not ck.get('complete'):raise ValueError(f'Incomplete training: {path}')
    return ck

def predict_model(model,Y,device='cpu',batch_size=256):
    model.eval();out=[]
    with torch.inference_mode():
        for start in range(0,len(Y),batch_size):
            y=torch.as_tensor(Y[start:start+batch_size],dtype=torch.float32,device=device)
            out.append(model(y).cpu().numpy())
    return np.concatenate(out)

def atomic_torch_save(value,path):
    tmp=Path(str(path)+'.tmp');torch.save(value,tmp);os.replace(tmp,path)

def train_decoder(method,fold_data,M,seed,config,run_dir,Y_train,Y_val,data_hash,device=None):
    device=device or ('cuda' if torch.cuda.is_available() else 'cpu')
    run_dir=Path(run_dir);run_dir.mkdir(parents=True,exist_ok=True)
    name=f'{method}_fold{fold_data["fold"]}_M{M}_seed{seed}'
    path=run_dir/(name+'.pt');last_path=run_dir/(name+'.resume.pt');hist_path=run_dir/(name+'.history.json')
    contract=checkpoint_contract(method,fold_data,M,seed,config,data_hash);fingerprint=contract_hash(contract)
    if path.exists():
        ck=load_verified_checkpoint(path,fingerprint,require_complete=False)
        if ck.get('complete'):print('REUSE',name,flush=True);return path
    set_seed(seed);model=build_decoder(method,M).to(device);opt=torch.optim.Adam(model.parameters(),lr=config.learning_rate,capturable=device=='cuda')
    x=torch.as_tensor(fold_data['train']['windows'],device=device);y=torch.as_tensor(Y_train,device=device)
    best=float('inf');best_epoch=0;best_state=None;stale=0;history=[];start_epoch=1;elapsed_prior=0.
    if last_path.exists():
        state=load_verified_checkpoint(last_path,fingerprint,False)
        model.load_state_dict(state['current_state']);opt.load_state_dict(state['optimizer_state'])
        best=state['val_macro_prd'];best_epoch=state['epoch'];best_state=state['model_state_dict'];stale=state['stale'];history=state['history']
        start_epoch=history[-1]['epoch']+1;elapsed_prior=state['training_seconds']
        torch.set_rng_state(state['torch_rng']);np.random.set_state(state['numpy_rng']);random.setstate(state['python_rng'])
        if device=='cuda':torch.cuda.set_rng_state_all(state['cuda_rng'])
    if device=='cuda':torch.cuda.reset_peak_memory_stats()
    captured=CapturedAdamStep(model,opt,y[:config.batch_size],x[:config.batch_size]) if device=='cuda' else None
    start=time.perf_counter()
    print('TRAIN',name,'windows',len(x),'val',len(Y_val),'device',device,flush=True)
    for epoch in range(start_epoch,config.max_epochs+1):
        if stale>=config.patience:break
        epoch_start=time.perf_counter();model.train();order=torch.randperm(len(x),device=device)
        loss_sum=torch.zeros((),device=device)
        for indices in order.split(config.batch_size):
            if captured is not None:loss=captured.step(y[indices],x[indices])
            else:
                opt.zero_grad(set_to_none=True);pred=model(y[indices]);loss=(pred-x[indices]).square().mean();loss.backward();opt.step()
            loss_sum+=loss.detach()*len(indices)
        if not torch.isfinite(loss_sum):raise FloatingPointError(f'Nonfinite training loss {name} epoch {epoch}')
        val_pred=predict_model(model,Y_val,device)
        score=compute_macro_prd(fold_data['val']['windows'],val_pred,fold_data['val']['subjects'])
        row={'epoch':epoch,'train_loss':float(loss_sum.cpu())/len(x),'val_macro_prd':score,'epoch_seconds':time.perf_counter()-epoch_start}
        history.append(row)
        if score<best:
            best=score;best_epoch=epoch;stale=0;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:stale+=1
        state={'complete':False,'fingerprint':fingerprint,'contract':contract,'model_state_dict':best_state,
               'val_macro_prd':best,'epoch':best_epoch,'history':history,'stale':stale,
               'training_seconds':elapsed_prior+time.perf_counter()-start,'parameter_count':count_parameters(model),
               'device':device,'gpu':torch.cuda.get_device_name() if device=='cuda' else None,
               'peak_gpu_bytes':torch.cuda.max_memory_allocated() if device=='cuda' else 0}
        save_json(hist_path,{k:v for k,v in state.items() if k!='model_state_dict'})
        atomic_torch_save(state,path)
        resume=dict(state,current_state=model.state_dict(),optimizer_state=opt.state_dict(),torch_rng=torch.get_rng_state(),
                    numpy_rng=np.random.get_state(),python_rng=random.getstate(),cuda_rng=torch.cuda.get_rng_state_all() if device=='cuda' else [])
        atomic_torch_save(resume,last_path)
        if epoch==1 or epoch%5==0:print(name,row,'best',best_epoch,'stale',stale,flush=True)
    state=load_verified_checkpoint(path,fingerprint,False);state['complete']=True
    state['stop_reason']='patience' if stale>=config.patience else 'max_epochs'
    atomic_torch_save(state,path);save_json(hist_path,{k:v for k,v in state.items() if k!='model_state_dict'})
    if last_path.exists():last_path.unlink()
    print('COMPLETE',name,'best',best,'epoch',best_epoch,'total_seconds',state['training_seconds'],flush=True)
    return path
