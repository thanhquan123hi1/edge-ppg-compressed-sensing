"""Physical raw-stream verification. Missing/failed hardware never supplies fake packets."""
import collections
import json
import struct
import time
from pathlib import Path
import numpy as np
import pandas as pd
import serial
from serial.tools import list_ports
from preprocess import CausalSOSFilter
from dataset import load_raw_bvp
from sensing import generate_rademacher_matrix,project_measurements
from quantization import quantize_int16,unpack_packet,pack_packet
from provenance import save_json,contract_hash,file_sha256

class ReferenceRawStream:
    def __init__(self,mu,sigma):
        self.mu=np.float32(mu);self.sigma=np.float32(sigma);self.filter=CausalSOSFilter();self.raw_count=0;self.ring=collections.deque(maxlen=256);self.aborted=False
    def feed(self,raw):
        if self.aborted:raise ValueError('Stream aborted; reset by creating a new session')
        raw=np.asarray(raw,dtype=np.float32)
        if not np.isfinite(raw).all():
            self.aborted=True;raise ValueError('Stream requires finite raw samples; reset after gap')
        filtered=self.filter.filter_chunk(raw);frames=[]
        for f in filtered:
            self.raw_count+=1
            if self.raw_count>320:self.ring.append(f)
            if self.raw_count>=576 and (self.raw_count-576)%128==0:
                values=np.asarray(self.ring,dtype=np.float32)
                frames.append({'raw_start_sample':self.raw_count-256,'filtered':values,'x':(values-self.mu)/self.sigma})
        return frames

class SequenceValidator:
    def __init__(self):self.next=0
    def check(self,index):
        if index!=self.next:raise ValueError(f'Wrong frame sequence {index}; expected {self.next}')
        self.next+=1

def read_exact(port,n,capture_path=None):
    value=bytearray();deadline=time.monotonic()+10
    while len(value)<n:
        part=port.read(n-len(value));value.extend(part)
        if part and capture_path is not None:
            with Path(capture_path).open('ab') as f:f.write(part)
        if time.monotonic()>deadline:raise TimeoutError(f'Serial short read {len(value)}/{n}')
    return bytes(value)

def connect(port):
    ser=serial.Serial(port,115200,timeout=.5,write_timeout=10);time.sleep(.8);ser.reset_input_buffer();return ser

def query_info(ser):
    ser.write(b'INFO\n');lines=[];deadline=time.monotonic()+10
    while time.monotonic()<deadline:
        line=ser.readline().decode('ascii',errors='replace').strip();lines.append(line)
        if line=='INFO_END':return dict(line.split('=',1) for line in lines if '=' in line)
    raise TimeoutError('INFO response incomplete')

def collect_session(ser,config,registry,config_id,subject,frames,trace,output_tag,warmup_frames=0,measurement_session_id=None):
    dest=config.directory('results')/'hardware';dest.mkdir(exist_ok=True);wire_path=dest/(output_tag+'.bin');wire_path.write_bytes(b'')
    cfg=registry[str(config_id)];count=576+128*(frames-1);raw=load_raw_bvp(subject,config.data_root)[:count]
    if len(raw)!=count:raise ValueError('Raw record too short')
    reference=ReferenceRawStream(cfg['mu'],cfg['sigma']);seq=SequenceValidator();packets=[];timings=[];traces=[]
    ser.write(f'STREAM_BEGIN {config_id} {count} {int(trace)}\n'.encode())
    ready=ser.readline().decode('ascii').strip()
    if ready!=f'READY {config_id} {count} {contract_hash({int(k):v for k,v in registry.items()})}':raise ValueError(f'Firmware registry/READY mismatch {ready}')
    raw_output=bytearray();decoded=[];reference_frames=[]
    _,phi=generate_rademacher_matrix(cfg['M'])
    # Send only newly arrived samples; first frame consumes 576, later frames 128.
    boundaries=[(0,576)]+[(576+i*128,704+i*128) for i in range(frames-1)]
    for frame_index,(lo,hi) in enumerate(boundaries):
        reference_frame=reference.feed(raw[lo:hi])[0];ser.write(raw[lo:hi].astype('<f4').tobytes())
        header=read_exact(ser,10,wire_path)
        if header[:4]!=b'CSF1':raise ValueError(f'UART framing mismatch {header!r}')
        kind,length=struct.unpack('<HI',header[4:]);expected_kind=2 if trace else 1
        expected_length=16+2*cfg['M']+36+((512+cfg['M'])*4 if trace else 0)
        if kind!=expected_kind or length!=expected_length:raise ValueError('UART version/type/length mismatch')
        body=read_exact(ser,length,wire_path);raw_output.extend(header+body);plen=16+2*cfg['M'];packet=body[:plen]
        try:
            decoded_packet=unpack_packet(packet,registry);seq.check(decoded_packet['frame_index'])
            if decoded_packet['config_id']!=config_id:raise ValueError('Wrong packet config')
        except Exception as exc:
            failed_dir=dest/'failures';failed_dir.mkdir(exist_ok=True);failed_path=failed_dir/f'{output_tag}_frame{frame_index}.npz'
            np.savez_compressed(failed_path,packet_bytes=np.frombuffer(packet,dtype='uint8'),reference_x=reference_frame['x'],
                subject=subject,raw_start_sample=reference_frame['raw_start_sample'],expected_frame_index=frame_index,expected_config_id=config_id)
            save_json(str(failed_path)+'.json',{'status':'failed','stage':'packet_receiver','reason':str(exc),'measurement_session_id':measurement_session_id})
            raise
        timing=struct.unpack('<9I',body[plen:plen+36]);fields=['raw_end_sample','filter_us','normalize_us','project_us','quantize_us','packet_us','encoder_us','internal_free_bytes','stack_free_bytes']
        timings.append(dict(zip(fields,timing),frame_index=frame_index,config_id=config_id,M=cfg['M'],subject=subject,trace=trace,
                            warmup=frame_index<warmup_frames,measurement_session_id=measurement_session_id))
        if timing[0]!=hi:raise ValueError('Hardware stream cadence mismatch')
        packets.append(np.frombuffer(packet,dtype='uint8').copy());decoded.append(decoded_packet['y_tilde']);reference_frames.append(reference_frame['x'])
        if trace:
            trace_values=np.frombuffer(body[plen+36:],dtype='<f4');ff,xx,yy=trace_values[:256],trace_values[256:512],trace_values[512:]
            py_y=project_measurements(reference_frame['x'],phi);py_q,py_delta,_,_=quantize_int16(py_y)
            py_packet=pack_packet(py_q,py_delta,frame_index,config_id)
            unpack_packet(py_packet,registry)
            x_ok=np.allclose(xx,reference_frame['x'],atol=1e-4,rtol=1e-4)
            y_ok=np.allclose(yy,py_y,atol=1e-4,rtol=1e-4)
            f_ok=np.allclose(ff,reference_frame['filtered'],atol=1e-4,rtol=1e-4)
            q_diff=int(np.max(np.abs(decoded_packet['q'].astype('int32')-py_q.astype('int32'))))
            traces.append(dict(frame_index=frame_index,subject=subject,config_id=config_id,M=cfg['M'],measurement_session_id=measurement_session_id,
                max_abs_filter=float(np.max(np.abs(ff-reference_frame['filtered']))),max_abs_x=float(np.max(np.abs(xx-reference_frame['x']))),
                max_abs_y=float(np.max(np.abs(yy-py_y))),max_q_lsb=q_diff,scale_abs_diff=abs(decoded_packet['delta']-py_delta),
                bytes_identical=packet==py_packet,crc_hardware_valid=True,crc_python_valid=True,
                status='exact' if packet==py_packet and np.array_equal(xx,reference_frame['x']) else 'tolerant' if x_ok and y_ok and f_ok and q_diff<=1 else 'failed'))
        if (frame_index+1)%200==0:print('HIL',output_tag,frame_index+1,'/',frames,flush=True)
    done=ser.readline().decode('ascii').strip()
    if done!=f'STREAM_DONE {count} {frames}':raise ValueError(f'Unexpected stream completion {done}')
    ser.write(b'STREAM_END\n')
    if ser.readline().strip()!=b'OK_END':raise ValueError('STREAM_END failed')
    (dest/(output_tag+'.bin')).write_bytes(raw_output)
    np.savez_compressed(dest/(output_tag+'.npz'),y_tilde=np.asarray(decoded),x_reference=np.asarray(reference_frames),
        packets=np.asarray(packets),raw_start_sample=np.arange(frames,dtype='int64')*128+320,
        subject=subject,config_id=config_id,measurement_session_id=measurement_session_id or 'manual',warmup=np.arange(frames)<warmup_frames,
        raw_input_sha256=__import__('hashlib').sha256(raw.astype('<f4').tobytes()).hexdigest())
    return timings,traces

def run_hil(config,frames=1000):
    directory=config.directory('results');status_path=directory/'hardware_status.json'
    if status_path.exists():
        import datetime,shutil
        archive=directory/'hardware'/'archives'/datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f');archive.mkdir(parents=True,exist_ok=True)
        for old in [status_path,directory/'hardware_timings.csv',directory/'hardware_traces.csv',directory/'ram_comparison.json']:
            if old.exists():shutil.copy2(old,archive/old.name)
        for old in (directory/'hardware').glob('*'):
            if old.is_file():shutil.copy2(old,archive/old.name)
    if config.hardware_port not in [p.device for p in list_ports.comports()]:
        result={'status':'skipped','reason':f'Port {config.hardware_port} absent','measured_frames':0};save_json(status_path,result)
        if config.hardware_required:raise RuntimeError(result['reason'])
        return result
    try:
        import datetime
        session_id=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');warmup=32
        save_json(status_path,{'status':'running','measurement_session_id':session_id,'warmup_per_M':warmup})
        registry=json.loads((config.directory('exports')/'config_registry.json').read_text(encoding='utf-8'));timings=[];traces=[]
        with connect(config.hardware_port) as ser:
            info_before=query_info(ser)
            if info_before['PROTOCOL_SHA']!=contract_hash(config.numerical_contract()):raise ValueError('Hardware protocol fingerprint mismatch')
            for fold in range(1,6):
                subject=f'S{(fold-1)*3+1}'
                for M in config.M_list:
                    tag=f'trace_fold{fold}_M{M}';t,tr=collect_session(ser,config,registry,fold*1000+M,subject,8,True,tag,measurement_session_id=session_id);timings.extend(t);traces.extend(tr)
            for M in config.M_list:
                t,tr=collect_session(ser,config,registry,1000+M,'S1',frames+warmup,False,f'benchmark_M{M}',warmup,session_id);timings.extend(t)
            info_after=query_info(ser)
        pd.DataFrame(timings).to_csv(directory/'hardware_timings.csv',index=False);pd.DataFrame(traces).to_csv(directory/'hardware_traces.csv',index=False)
        measured=[r for r in timings if not r['trace'] and not r['warmup']];failed=[r for r in traces if r['status']=='failed']
        result={'status':'measured' if not failed else 'failed','measured_frames':len(measured),'trace_frames':len(traces),'trace_failures':len(failed),
                'crc_rejections':0,'missing_frames':0,'duplicate_frames':0,'info_before':info_before,'info_after':info_after,
                'measurement_session_id':session_id,'warmup_per_M':warmup,'warmup_frames':3*warmup,
                'artifact_sha256':{p.name:file_sha256(p) for p in [directory/'hardware_timings.csv',directory/'hardware_traces.csv']+
                                  [p for p in (directory/'hardware').glob('*') if p.is_file() and p.stem.startswith(('trace_fold','benchmark_M')) and '_prediction_' not in p.stem]},
                'port':config.hardware_port,'baud':115200,'tolerance':{'atol':1e-4,'rtol':1e-4,'q_lsb':1},
                'firmware_source_sha256':file_sha256(Path(config.root)/'esp32_firmware/src/main.cpp'),
                'timing_note':'encoder_us: after normalized ring window to completed packet/CRC, excluding filter, normalization and UART; filter_us includes per-sample timer overhead; 32 warmup frames per M excluded from benchmark.'}
        save_json(status_path,result)
        if failed:raise RuntimeError(f'{len(failed)} physical trace tolerance failures')
        return result
    except Exception as exc:
        previous=json.loads(status_path.read_text(encoding='utf-8')) if status_path.exists() else {}
        if previous.get('status')=='failed' and previous.get('measurement_session_id')==locals().get('session_id'):
            result=dict(previous,reason=str(exc))
        else:
            records=locals().get('timings',[])
            result={'status':'failed','reason':str(exc),'measurement_session_id':locals().get('session_id'),
                'measured_frames':sum(not r['trace'] and not r.get('warmup',False) for r in records),'crc_rejections':int('CRC32 mismatch' in str(exc))}
        save_json(status_path,result);raise

if __name__=='__main__':
    from experiment_config import ExperimentConfig
    print(run_hil(ExperimentConfig()))
