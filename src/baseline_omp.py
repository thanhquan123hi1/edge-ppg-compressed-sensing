"""OMP-DCT with twice-reorthogonalized incremental QR and full validation paths."""
import numpy as np
import scipy.fft
import scipy.linalg
from evaluate import compute_window_metrics,aggregate_subject_metrics,compute_macro_statistics

def build_dct_matrix(N=256):
    return scipy.fft.dct(np.eye(N,dtype=np.float32),type=2,norm='ortho',axis=0)

class OMPDecoder:
    def __init__(self,Phi,N=256):
        self.Phi=np.asarray(Phi,dtype=np.float32);self.M,self.N=self.Phi.shape
        if self.N!=N:raise ValueError('Wrong N')
        self.C=build_dct_matrix(N);self.Psi=self.C.T.copy()
        self.A=(self.Phi@self.Psi).astype(np.float32)
        norms=np.linalg.norm(self.A,axis=0).astype(np.float32)
        self.available=norms>=1e-12
        self.d=np.where(self.available,norms,1).astype(np.float32)
        self.A_bar=(self.A/self.d[None,:]).astype(np.float32)
        self.dropped_columns=int(np.sum(~self.available))

    def decode_path(self,y_tilde,candidates,tol=1e-6):
        y=np.asarray(y_tilde,dtype=np.float32)
        if y.shape!=(self.M,) or not np.isfinite(y).all():raise ValueError('Invalid measurement')
        candidates=sorted(set(int(k) for k in candidates))
        if not candidates or candidates[0]<=0 or candidates[-1]>=self.M:raise ValueError('Require 0<K<M')
        limit=candidates[-1]; threshold=tol*max(float(np.linalg.norm(y)),1.)
        Q=np.zeros((self.M,limit),dtype=np.float32);R=np.zeros((limit,limit),dtype=np.float32)
        rhs=np.zeros(limit,dtype=np.float32);selected=[];r=y.copy();outputs={}
        reason='max_iter';last=np.zeros(self.N,dtype=np.float32)
        if float(np.linalg.norm(y))<=threshold:
            return {k:(last.copy(),'small_measurement',0) for k in candidates}
        for step in range(limit):
            corr=np.abs(self.A_bar.T@r);corr[~self.available]=-np.inf
            if selected:corr[selected]=-np.inf
            j=int(np.argmax(corr))
            if not np.isfinite(corr[j]):reason='no_more_atoms';break
            atom=self.A_bar[:,j];v=atom.copy()
            if step:
                h=Q[:,:step].T@v;v-=Q[:,:step]@h
                correction=Q[:,:step].T@v;h+=correction;v-=Q[:,:step]@correction
                R[:step,step]=h
            diagonal=float(np.linalg.norm(v))
            if diagonal<=1e-7:reason='rank_deficient';break
            selected.append(j);Q[:,step]=v/diagonal;R[step,step]=diagonal
            rhs[step]=Q[:,step]@y
            r=y-Q[:,:step+1]@rhs[:step+1]
            stop=float(np.linalg.norm(r))<=threshold
            if step+1 in candidates or stop or step+1==limit:
                beta=scipy.linalg.solve_triangular(R[:step+1,:step+1],rhs[:step+1],check_finite=False)
                alpha=np.zeros(self.N,dtype=np.float32);alpha[selected]=beta/self.d[selected]
                last=(self.Psi@alpha).astype(np.float32)
                if not np.isfinite(last).all():reason='nonfinite_solution';break
                if step+1 in candidates:outputs[step+1]=(last.copy(),'max_iter',step+1)
            if stop:reason='residual_tol';break
        if reason!='max_iter':
            for k in candidates:
                if k>=len(selected):outputs[k]=(last.copy(),reason,len(selected))
        for k in candidates:
            if k not in outputs:outputs[k]=(last.copy(),reason,len(selected))
        return outputs

    def decode_single(self,y_tilde,K_max=32,tol=1e-6,track_stats=False):
        if K_max<=0:
            result=(np.zeros(self.N,dtype=np.float32),'zero_iterations',0)
        else:result=self.decode_path(y_tilde,[K_max],tol)[K_max]
        return result if track_stats else result[0]

    def decode_batch(self,Y_tilde,K_max=32,tol=1e-6,track_stats=False):
        predictions=[];reasons=[];iterations=[]
        for y in Y_tilde:
            x,reason,n=self.decode_single(y,K_max,tol,True)
            predictions.append(x);reasons.append(reason);iterations.append(n)
        p=np.asarray(predictions,dtype=np.float32)
        return (p,np.asarray(reasons),np.asarray(iterations)) if track_stats else p

def get_candidate_k_max(M):
    return [k for k in [8,16,32,64] if k<M]

def select_best_k_max(omp_decoder,Y_val,X_val,val_subjects,M,tol_tie=1e-6,max_val_windows=None,failure_callback=None):
    # Deliberately use all validation windows. Path reuses the same QR trajectory.
    candidates=get_candidate_k_max(M)
    prds={k:np.empty(len(Y_val),dtype=np.float64) for k in candidates}
    valid=np.sum(X_val.astype(np.float64)**2,axis=1)>=1e-12
    for i,y in enumerate(Y_val):
        path=omp_decoder.decode_path(y,candidates)
        for k,(pred,reason,_) in path.items():
            if reason in ('rank_deficient','nonfinite_solution'):
                if failure_callback is not None:failure_callback(i,k,reason,pred)
                raise RuntimeError(f'OMP validation failure {reason} at window {i}, K{k}')
            prds[k][i]=100*np.linalg.norm(X_val[i].astype(np.float64)-pred)/np.linalg.norm(X_val[i].astype(np.float64)) if valid[i] else np.nan
    scores={k:float(np.mean([np.mean(prds[k][(val_subjects==s)&valid]) for s in np.unique(val_subjects)])) for k in candidates}
    best=min(scores.values());chosen=min(k for k in candidates if scores[k]<=best+tol_tie)
    return chosen,scores[chosen],scores
