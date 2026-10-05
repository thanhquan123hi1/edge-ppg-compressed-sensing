"""Measured train-only initialization audit and explicit application-byte math."""
import numpy as np
import pandas as pd
import scipy.signal
from experiment_config import filter_sos


def finite_stream_accounting(length,M,N=256,H=128,header=16):
    if length<N:raise ValueError('Signal shorter than first window')
    frames=1+(length-N)//H;used=N+(frames-1)*H
    raw=2*used+header*frames;compressed=frames*(header+2*M)
    return {'frames':frames,'unique_samples':used,'discarded_tail_samples':length-used,
            'raw_app_bytes':raw,'compressed_app_bytes':compressed,'finite_CR':raw/compressed,
            'scope':'derived app bytes; INT16 first N then H new samples; excludes unmeasured radio overhead'}


def audit_initialization(raw_dataset,train_subjects,starts_seconds=(60,120,180),observe_seconds=30):
    """Reset at known train times versus a continuously running filter on real input.

    This post-experiment audit does not retroactively justify or change drop=320.
    The declared diagnostic criterion is <=1% PRD on windows after 5 seconds.
    """
    sos=filter_sos();rows=[]
    for subject in train_subjects:
        raw=np.asarray(raw_dataset[subject],dtype=np.float32)
        for second in starts_seconds:
            start=int(second*64);stop=start+int(observe_seconds*64)
            if stop>len(raw):raise ValueError('Train segment too short for initialization audit')
            prefix=raw[:start];segment=raw[start:stop]
            if not np.isfinite(prefix).all() or not np.isfinite(segment).all():raise ValueError('Nonfinite train audit input')
            _,state=scipy.signal.sosfilt(sos,prefix,zi=np.zeros((2,2),dtype=np.float32))
            reference=scipy.signal.sosfilt(sos,segment,zi=state)[0]
            reset=scipy.signal.sosfilt(sos,segment,zi=np.zeros((2,2),dtype=np.float32))[0]
            ref=reference[320:576].astype(np.float64);error=(reset-reference)[320:576].astype(np.float64)
            energy=np.sum(ref**2)
            prd=100*np.sqrt(np.sum(error**2)/energy) if energy>=1e-12 else np.nan
            rows.append({'Subject':subject,'raw_reset_sample':start,'reference_window_start_seconds':5.,
                         'prd_after_5s':prd,'max_abs_error_after_5s':float(np.max(np.abs(error))),
                         'status':'UNDEFINED_ZERO_REFERENCE' if not np.isfinite(prd) else 'PASS_1PCT_DIAGNOSTIC' if prd<=1 else 'FAIL_1PCT_DIAGNOSTIC',
                         'scope':'post-experiment train-only reset audit; does not tune drop'})
    return pd.DataFrame(rows)


def summarize_host_timings(hardware_result):
    frame=pd.DataFrame(hardware_result['timing_records'])
    frame=frame[~frame.trace & ~frame.warmup]
    rows=[]
    for M,part in frame.groupby('M'):
        for column,label in [('filter_us','Filter newly received samples'),('normalize_us','Normalize window'),
                ('encoder_us','Projection+quantization+CRC/packet'),('host_raw_chunk_to_packet_ms','Host raw write→packet received'),
                ('host_lite_decoder_ms','Host Lite decoder'),('host_raw_chunk_to_lite_ms','Host raw write→Lite output')]:
            values=part[column].to_numpy(dtype=float)/(1000 if column.endswith('_us') else 1)
            if not np.isfinite(values).all() or (values<0).any():raise ValueError('Invalid measured timing')
            rows.append({'M':int(M),'Stage':label,'n':len(values),'mean_ms':float(values.mean()),
                         'p50_ms':float(np.percentile(values,50)),'p95_ms':float(np.percentile(values,95)),
                         'measurement_session_id':hardware_result['measurement_session_id']})
    return pd.DataFrame(rows)
