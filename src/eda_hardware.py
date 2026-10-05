"""Measured raw-stream HIL for EDA. Never use Python packets as device input.

The serial protocol is the inspected CSF1 streaming firmware. The raw dataset
argument is checked against signal/wrist/BVP in each original subject pickle.
Logs and detailed arrays are saved internally; this module prints no paths.
"""
import contextlib
import datetime
import hashlib
import io
import json
import struct
import time
import uuid
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from dataset import load_raw_bvp as _eda_load_raw_bvp
from provenance import contract_hash as _eda_contract_hash, file_sha256 as _eda_file_sha256, save_json as _eda_save_json
from quantization import pack_packet as _eda_pack_packet, quantize_int16 as _eda_quantize_int16, unpack_packet as _eda_unpack_packet
from sensing import generate_rademacher_matrix as _eda_generate_matrix, project_measurements as _eda_project
from verify_esp32 import ReferenceRawStream as _EDAReferenceRawStream, connect as _eda_connect, query_info as _eda_query_info


EDA_HARDWARE_STAGES = ('raw input identity', 'filtered', 'normalized', 'y', 'scale', 'q_int16',
                       'payload bytes', 'packet bytes', 'CRC validity', 'packet size',
                       'unknown config_id rejection', 'CRC common byte vector')


def eda_crc32(value):
    """CRC-32/ISO-HDLC: reflected EDB88320, init/final xor FFFFFFFF."""
    return zlib.crc32(value) & 0xffffffff


def verify_eda_crc_vector(serial_port, value):
    """Device challenge on echoed bytes; missing support remains unmeasured."""
    value=bytes(value)
    if not 1<=len(value)<=64:raise ValueError('CRC challenge requires 1..64 bytes')
    serial_port.write(b'CRC32 '+value.hex().encode()+b'\n')
    response=serial_port.readline().decode('ascii',errors='replace').strip()
    tokens=response.split()
    measured=len(tokens)==3 and tokens[0]=='CRC32'
    equal=False;device_crc=None;echo=None
    if measured:
        try:
            echo=bytes.fromhex(tokens[1]);device_crc=int(tokens[2],16)
            equal=echo==value and device_crc==eda_crc32(value) and len(tokens[2])==8
        except ValueError:pass
    return {'stage':'CRC common byte vector','status':'PASS' if equal else 'FAIL' if measured else 'NOT_MEASURED',
            'exact_equal':equal if measured else None,'python_crc':eda_crc32(value),
            'device_crc':device_crc,'input_hex':value.hex(),'device_echo_hex':echo.hex() if echo is not None else None,
            'detail':'Same-byte CRC-32/ISO-HDLC challenge; echoed bytes must match' if measured else 'Firmware CRC challenge unavailable'}


def inspect_eda_packet(packet, registry, reference_packet=None):
    """Reject before any decoder; CRC validity and byte equality are separate."""
    packet = bytes(packet)
    decoded = _eda_unpack_packet(packet, registry)
    if decoded['flags'] != 0:
        raise ValueError('Raw HIL requires INT16 packet flags=0')
    # Recompute from the received bytes, excluding only the stored CRC field.
    crc_actual = eda_crc32(packet[:12] + packet[16:])
    crc_stored = struct.unpack('<I', packet[12:16])[0]
    result = dict(decoded, crc_valid=crc_actual == crc_stored, crc_recomputed=crc_actual,
                  byte_equal=None, reference_crc_valid=None, reference_crc32=None)
    if reference_packet is not None:
        reference = inspect_eda_packet(reference_packet, registry)
        result.update(byte_equal=packet == bytes(reference_packet),
                      reference_crc_valid=reference['crc_valid'], reference_crc32=reference['crc32'])
    return result


def test_eda_packet_receiver(received_packet, registry):
    """Adversarial receiver tests derived from one valid actual received packet."""
    inspect_eda_packet(received_packet,registry)
    fields=list(struct.unpack('<BBHIf',bytes(received_packet)[:12]));payload=bytes(received_packet)[16:]
    cases=[]
    for label,index,value in [('unknown config_id',2,9999),('unsupported version',0,2),
                               ('unsupported flags',1,2),('nonpositive scale',4,0.),('nonfinite scale',4,float('nan'))]:
        changed=fields.copy();changed[index]=value
        header=struct.pack('<BBHIf',*changed)
        cases.append((label,header+struct.pack('<I',eda_crc32(header+payload))+payload,True))
    header=bytes(received_packet)[:12];short_payload=payload[:-2]
    cases.append(('wrong payload length',header+struct.pack('<I',eda_crc32(header+short_payload))+short_payload,True))
    bad=bytearray(received_packet);bad[-1]^=1
    cases.append(('invalid CRC',bytes(bad),False))
    rows=[]
    for label,mutated,crc_valid in cases:
        try:
            inspect_eda_packet(mutated,registry);actual='ACCEPT'
        except ValueError:actual='REJECT'
        rows.append({'test':label,'expected':'REJECT','actual':actual,'status':'PASS' if actual=='REJECT' else 'FAIL',
                     'mutated_crc_valid':crc_valid,'source':'Mutation of actual received packet; receiver test, no decoder'})
    return pd.DataFrame(rows)


def audit_eda_raw(raw_bvp_dataset, subject, sample_count, data_root):
    """Require complete original raw arrays, then replay prefix including transient."""
    if subject not in raw_bvp_dataset:
        raise ValueError('Missing raw BVP subject')
    supplied = np.asarray(raw_bvp_dataset[subject], dtype=np.float32)
    source = _eda_load_raw_bvp(subject, data_root)
    if supplied.ndim != 1 or supplied.shape != source.shape or not np.array_equal(supplied, source, equal_nan=True):
        raise ValueError('Input differs from original raw source BVP; filtered/normalized/truncated arrays are forbidden')
    if sample_count < 576 or len(supplied) < sample_count:
        raise ValueError('Raw source record too short for requested frames')
    selected = np.ascontiguousarray(supplied[:sample_count], dtype='<f4')
    if not np.isfinite(selected).all():
        raise ValueError('Raw replay requires finite continuous samples; no silent gap removal')
    return selected, {'subject':subject, 'identity_equal':True, 'raw_start_sample':0,
                      'raw_samples':len(source), 'replayed_samples':sample_count,
                      'transient_sent_samples':320,
                      'source_sha256':_eda_file_sha256(Path(data_root)/subject/f'{subject}.pkl'),
                      'raw_array_sha256':hashlib.sha256(supplied.astype('<f4').tobytes()).hexdigest(),
                      'replayed_sha256':hashlib.sha256(selected.tobytes()).hexdigest()}


def compare_eda_trace(reference, trace_values, packet, registry, metadata):
    """Compare independently received trace arrays to Float32 causal reference."""
    observed = inspect_eda_packet(packet, registry, reference['packet'])
    M = len(observed['q'])
    trace_values = np.asarray(trace_values, dtype=np.float32)
    if trace_values.shape != (512+M,) or not np.isfinite(trace_values).all():
        raise ValueError('Invalid trace shape/nonfinite values')
    actual = {'filtered':trace_values[:256], 'normalized':trace_values[256:512],
              'y':trace_values[512:], 'scale':np.asarray([observed['delta']], dtype=np.float32)}
    expected = {'filtered':reference['filtered'], 'normalized':reference['x'],
                'y':reference['y'], 'scale':np.asarray([reference['scale']], dtype=np.float32)}
    rows = []
    for stage in actual:
        difference = actual[stage].astype(np.float64)-np.asarray(expected[stage],dtype=np.float64)
        exact = bool(np.array_equal(actual[stage], expected[stage]))
        close = bool(np.allclose(actual[stage],expected[stage],atol=1e-4,rtol=1e-4))
        rows.append(dict(metadata, stage=stage, max_abs_error=float(np.max(np.abs(difference))),
                         rmse=float(np.sqrt(np.mean(difference*difference))), exact_equal=exact,
                         max_lsb_diff=np.nan, status='PASS' if close else 'FAIL',
                         detail='Float32 tolerance atol=rtol=1e-4'))
    q_difference = observed['q'].astype(np.int32)-np.asarray(reference['q'],dtype=np.int32)
    max_lsb = int(np.max(np.abs(q_difference)))
    exact_q = bool(np.array_equal(observed['q'], reference['q']))
    rows.append(dict(metadata, stage='q_int16', max_abs_error=max_lsb,
                     rmse=float(np.sqrt(np.mean(q_difference.astype(np.float64)**2))),
                     max_lsb_diff=max_lsb, exact_equal=exact_q,
                     status='PASS' if exact_q else 'WITHIN_1_LSB' if max_lsb<=1 else 'FAIL',
                     detail='Bit-exact INT16' if exact_q else 'Numerical agreement within 1 LSB' if max_lsb<=1 else 'INT16 mismatch'))
    for stage, valid in [('payload bytes',bytes(packet)[16:]==reference['packet'][16:]),
                         ('packet bytes',observed['byte_equal']),
                         ('CRC validity',observed['crc_valid'] and observed['reference_crc_valid']),
                         ('packet size',len(packet)==16+2*M)]:
        rows.append(dict(metadata, stage=stage, max_abs_error=np.nan, rmse=np.nan,
                         max_lsb_diff=np.nan, exact_equal=bool(valid) if stage!='CRC validity' else None,
                         status='PASS' if valid else 'FAIL',
                         detail='CRC recalculated independently on each header[0:12]+payload' if stage=='CRC validity' else 'Byte comparison'))
    record = dict(metadata, packet_bytes=bytes(packet), packet_python=reference['packet'],
                  x_reference=np.asarray(reference['x']).copy(), filtered_reference=np.asarray(reference['filtered']).copy(),
                  y_python=np.asarray(reference['y']).copy(), q_python=np.asarray(reference['q']).copy(),
                  scale_python=reference['scale'], filtered_esp32=actual['filtered'].copy(),
                  x_esp32=actual['normalized'].copy(), y_esp32=actual['y'].copy(),
                  q_esp32=observed['q'].copy(), scale_esp32=observed['delta'],
                  y_tilde=observed['y_tilde'].copy(), crc_valid=observed['crc_valid'],
                  byte_equal=observed['byte_equal'], max_q_lsb=max_lsb)
    return rows, record


def summarize_eda_timing(records, M_list=(51,77,128)):
    rows=[]
    for M in M_list:
        values=np.asarray([r['encoder_us']/1000. for r in records
                           if r['M']==M and not r.get('trace',False) and not r.get('warmup',False)],dtype=np.float64)
        if not np.isfinite(values).all() or np.any(values<0):
            raise ValueError('Invalid encoder device timings')
        rows.append({'M':M,'n':len(values),'mean_ms':float(values.mean()) if len(values) else np.nan,
                     'p50_ms':float(np.percentile(values,50)) if len(values) else np.nan,
                     'p95_ms':float(np.percentile(values,95)) if len(values) else np.nan,
                     'min_ms':float(values.min()) if len(values) else np.nan,
                     'max_ms':float(values.max()) if len(values) else np.nan,
                     'status':'MEASURED' if len(values) else 'NOT_MEASURED'})
    return pd.DataFrame(rows)


def summarize_eda_ram(info, comparison=None):
    mappings={'internal_free':'INTERNAL_FREE','internal_min_free':'INTERNAL_MIN_FREE',
              'internal_largest_free_block':'INTERNAL_LARGEST','stack_min_free':'STACK_FREE_BYTES',
              'PSRAM_capacity':'PSRAM_BYTES','PSRAM_free':'PSRAM_FREE',
              'application_static_buffers':'APP_BUFFER_BYTES','matrix_flash':'MATRICES_FLASH_BYTES'}
    rows=[{'metric':metric,'value_bytes':int(info[key]) if key in info else np.nan,
           'status':'MEASURED' if key in info else 'NOT_MEASURED',
           'detail':'Device INFO; stack is minimum unused task stack' if metric=='stack_min_free' else 'Device INFO'}
          for metric,key in mappings.items()]
    measured=comparison is not None and comparison.get('status')=='measured'
    for metric,value in [('full_static_dram',comparison.get('full_elf',{}).get('static_dram_bytes') if measured else None),
                         ('baseline_static_dram',comparison.get('baseline_elf',{}).get('static_dram_bytes') if measured else None),
                         ('additional_static_dram',comparison.get('incremental_static_bytes') if measured else None),
                         ('additional_internal_peak',comparison.get('incremental_peak_reserved_internal_bytes') if measured else None)]:
        rows.append({'metric':metric,'value_bytes':value if value is not None else np.nan,
                     'status':'MEASURED' if value is not None else 'NOT_MEASURED',
                     'detail':'Matched baseline measurement/build; no double-counting static+heap+stack'})
    return pd.DataFrame(rows)


class _EDASerialCapture:
    """Capture all transmitted/received bytes with an offset journal, no display."""
    def __init__(self,serial_port,directory):
        self.port=serial_port;self.directory=Path(directory);self.offset={'tx':0,'rx':0}
        for direction in self.offset:
            (self.directory/f'{direction}.bin').write_bytes(b'')

    def _record(self,direction,value,started_ns,finished_ns):
        value=bytes(value)
        if value:
            with (self.directory/f'{direction}.bin').open('ab') as f:f.write(value)
            with (self.directory/'serial_journal.jsonl').open('a',encoding='utf-8') as f:
                f.write(json.dumps({'direction':direction,'offset':self.offset[direction],
                                    'length':len(value),'monotonic_s':time.monotonic(),
                                    'io_start_perf_counter_ns':started_ns,
                                    'io_end_perf_counter_ns':finished_ns})+'\n')
            self.offset[direction]+=len(value)
        return value

    def write(self,value):
        started=time.perf_counter_ns()
        count=self.port.write(value)
        finished=time.perf_counter_ns()
        self._record('tx',value[:count],started,finished)
        if count!=len(value):raise IOError('Serial partial write')
        return count

    def read(self,count):
        started=time.perf_counter_ns();value=self.port.read(count);finished=time.perf_counter_ns()
        return self._record('rx',value,started,finished)
    def readline(self):
        started=time.perf_counter_ns();value=self.port.readline();finished=time.perf_counter_ns()
        return self._record('rx',value,started,finished)


def _eda_read_exact(serial_port,count):
    result=bytearray();deadline=time.monotonic()+10
    while len(result)<count:
        result.extend(serial_port.read(count-len(result)))
        if time.monotonic()>deadline:raise TimeoutError('Serial response incomplete')
    return bytes(result)


def _eda_validate_registry(config,registry):
    expected={str(fold*1000+M) for fold in range(1,6) for M in config.M_list}
    if set(map(str,registry))!=expected:raise ValueError('Registry must contain all 15 locked fold/M configurations')
    for key,cfg in registry.items():
        if int(key)!=int(cfg['fold'])*1000+int(cfg['M']) or cfg['M'] not in config.M_list:
            raise ValueError('Registry config_id/fold/M mismatch')
        if cfg.get('matrix_seed')!=config.matrix_seed or cfg.get('protocol_hash')!=_eda_contract_hash(config.numerical_contract()):
            raise ValueError('Registry numerical contract mismatch')
        if not np.isfinite(cfg['mu']) or not np.isfinite(cfg['sigma']) or cfg['sigma']<=0:
            raise ValueError('Invalid registry normalization')


def _eda_collect_session(serial_port,registry,config_id,subject,raw,frames,trace,warmup,session_id,
                         timings,stages,packets,directory,decoder_callback=None):
    cfg=registry[str(config_id)];count=576+128*(frames-1)
    registry_hash=_eda_contract_hash({int(k):v for k,v in registry.items()})
    serial_port.write(f'STREAM_BEGIN {config_id} {count} {int(trace)}\n'.encode())
    if serial_port.readline().strip()!=f'READY {config_id} {count} {registry_hash}'.encode():
        raise ValueError('Firmware registry/READY mismatch')
    reference=_EDAReferenceRawStream(cfg['mu'],cfg['sigma'])
    _,phi=_eda_generate_matrix(cfg['M'],seed=cfg['matrix_seed'])
    wire=bytearray();start_record=len(packets)
    for frame_index in range(frames):
        lo,hi=(0,576) if frame_index==0 else (576+128*(frame_index-1),576+128*frame_index)
        reference_frames=reference.feed(raw[lo:hi])
        if len(reference_frames)!=1:raise ValueError('Raw reference cadence mismatch')
        ref=reference_frames[0]
        raw_bytes=raw[lo:hi].astype('<f4').tobytes()
        host_started_ns=time.perf_counter_ns()
        serial_port.write(raw_bytes)
        header=_eda_read_exact(serial_port,10)
        if header[:4]!=b'CSF1':raise ValueError('UART framing mismatch')
        kind,length=struct.unpack('<HI',header[4:])
        expected_length=16+2*cfg['M']+36+((512+cfg['M'])*4 if trace else 0)
        if kind!=(2 if trace else 1) or length!=expected_length:raise ValueError('UART version/type/length mismatch')
        body=_eda_read_exact(serial_port,length)
        host_received_ns=time.perf_counter_ns()
        wire.extend(header+body)
        plen=16+2*cfg['M'];packet=body[:plen]
        decoded=inspect_eda_packet(packet,registry)
        host_lite_prediction=None
        if decoder_callback is not None:
            decoder_started_ns=time.perf_counter_ns()
            host_lite_prediction=np.asarray(decoder_callback(config_id,decoded['y_tilde']),dtype=np.float32)
            decoder_finished_ns=time.perf_counter_ns()
            if host_lite_prediction.shape!=(256,) or not np.isfinite(host_lite_prediction).all():
                raise ValueError('Host decoder callback returned invalid waveform')
        if decoded['config_id']!=config_id or decoded['frame_index']!=frame_index:
            raise ValueError('Wrong packet config/frame sequence')
        timing_fields=('raw_end_sample','filter_us','normalize_us','project_us','quantize_us','packet_us',
                       'encoder_us','internal_free_bytes','stack_free_bytes')
        tm=dict(zip(timing_fields,struct.unpack('<9I',body[plen:plen+36])))
        if tm['raw_end_sample']!=hi:raise ValueError('Hardware raw stream cadence mismatch')
        metadata={'subject':subject,'fold':cfg['fold'],'M':cfg['M'],'config_id':config_id,
                  'frame_index':frame_index,'raw_start_sample':hi-256,'trace':trace,
                  'warmup':frame_index<warmup,'measurement_session_id':session_id}
        metadata.update(host_raw_chunk_to_packet_ms=(host_received_ns-host_started_ns)/1e6,
                        host_raw_chunk_bytes=len(raw_bytes),host_response_bytes=10+length)
        if host_lite_prediction is not None:
            metadata.update(host_raw_chunk_to_lite_ms=(decoder_finished_ns-host_started_ns)/1e6,
                            host_lite_decoder_ms=(decoder_finished_ns-decoder_started_ns)/1e6)
        timings.append(dict(metadata,**tm))
        if trace:
            py_y=_eda_project(ref['x'],phi);py_q,py_scale,_,_=_eda_quantize_int16(py_y)
            ref.update(y=py_y,q=py_q,scale=py_scale,packet=_eda_pack_packet(py_q,py_scale,frame_index,config_id))
            rows,record=compare_eda_trace(ref,np.frombuffer(body[plen+36:],dtype='<f4'),packet,registry,metadata)
            stages.extend(rows)
        else:
            record=dict(metadata,packet_bytes=packet,x_reference=ref['x'].copy(),
                        y_tilde=decoded['y_tilde'].copy(),q_esp32=decoded['q'].copy(),
                        scale_esp32=decoded['delta'],crc_valid=decoded['crc_valid'])
        packets.append(record)
        if host_lite_prediction is not None:record['host_lite_prediction']=host_lite_prediction
    if serial_port.readline().strip()!=f'STREAM_DONE {count} {frames}'.encode():raise ValueError('Stream completion mismatch')
    serial_port.write(b'STREAM_END\n')
    if serial_port.readline().strip()!=b'OK_END':raise ValueError('STREAM_END acknowledgement missing')
    tag=f'{"trace" if trace else "benchmark"}_fold{cfg["fold"]}_M{cfg["M"]}'
    (Path(directory)/(tag+'.bin')).write_bytes(wire)
    records=packets[start_record:]
    arrays={'packets':np.asarray([np.frombuffer(r['packet_bytes'],dtype=np.uint8) for r in records]),
            'x_reference':np.asarray([r['x_reference'] for r in records]),
            'y_tilde':np.asarray([r['y_tilde'] for r in records]),
            'q_esp32':np.asarray([r['q_esp32'] for r in records]),
            'scale_esp32':np.asarray([r['scale_esp32'] for r in records]),
            'raw_start_sample':np.asarray([r['raw_start_sample'] for r in records]),
            'warmup':np.asarray([r['warmup'] for r in records]),
            'subject':subject,'config_id':config_id,'measurement_session_id':session_id}
    if trace:
        for name in ('filtered_reference','y_python','q_python','scale_python','filtered_esp32','x_esp32','y_esp32'):
            arrays[name]=np.asarray([r[name] for r in records])
    if decoder_callback is not None:
        arrays['host_lite_prediction']=np.asarray([r['host_lite_prediction'] for r in records])
    np.savez_compressed(Path(directory)/(tag+'.npz'),**arrays)


def _eda_empty_stages():
    return pd.DataFrame([{'stage':stage,'n':0,'max_abs_error':np.nan,'rmse':np.nan,
                          'max_lsb_diff':np.nan,'exact_equal':None,'status':'NOT_MEASURED',
                          'detail':'Chưa có phép đo thực tế'} for stage in EDA_HARDWARE_STAGES])


def _eda_stage_table(rows):
    if not rows:return _eda_empty_stages()
    frame=pd.DataFrame(rows);aggregated=[]
    for stage in EDA_HARDWARE_STAGES:
        subset=frame[frame['stage']==stage]
        if subset.empty:
            aggregated.extend(_eda_empty_stages().query('stage == @stage').to_dict('records'));continue
        statuses=set(subset['status'])
        status='FAIL' if 'FAIL' in statuses else 'NOT_MEASURED' if 'NOT_MEASURED' in statuses else 'WITHIN_1_LSB' if 'WITHIN_1_LSB' in statuses else 'PASS'
        exact=subset['exact_equal'].dropna() if 'exact_equal' in subset else pd.Series(dtype=bool)
        def maximum(name):
            values=pd.to_numeric(subset[name],errors='coerce').dropna() if name in subset else pd.Series(dtype=float)
            return float(values.max()) if len(values) else np.nan
        aggregated.append({'stage':stage,'n':len(subset),'max_abs_error':maximum('max_abs_error'),
                           'rmse':maximum('rmse'),'max_lsb_diff':maximum('max_lsb_diff'),
                           'exact_equal':bool(exact.all()) if len(exact) else None,'status':status,
                           'detail':'; '.join(dict.fromkeys(subset['detail'].dropna()))})
    return pd.DataFrame(aggregated)


def derive_eda_hardware_acceptance(result,stage_table,timing_table,ram_table,packets,frames):
    """Independent criteria; never infer one measurement from another."""
    stages=stage_table.set_index('stage')
    def stage_status(stage):return stages.loc[stage,'status'] if stage in stages.index else 'NOT_MEASURED'
    counts=timing_table['n'].to_numpy()
    timing_complete=bool(np.all(counts==frames)) and result.get('collection_complete',False)
    timing_status=('PASS' if bool((timing_table['p95_ms']<100.).all()) else 'FAIL') if timing_complete else 'PARTIAL' if counts.sum() else 'NOT_MEASURED'
    ram=ram_table.set_index('metric').loc['additional_internal_peak','value_bytes']
    ram_valid=result.get('ram_measurement_complete',False) and np.isfinite(ram)
    quant=stage_status('q_int16')
    quant_n=int(stages.loc['q_int16','n']) if 'q_int16' in stages.index else 0
    quant_complete=quant_n==120
    quant_status='PASS' if quant_complete and quant in ('PASS','WITHIN_1_LSB') else 'FAIL' if quant=='FAIL' else 'PARTIAL' if quant_n else 'NOT_MEASURED'
    crc_valid=bool(packets) and all(r.get('crc_valid',False) for r in packets)
    packet_complete=result.get('collection_complete',False) and len(packets)==120+3*(frames+32)
    packet_status='PASS' if packet_complete and crc_valid else 'FAIL' if packets and not crc_valid else 'PARTIAL' if packets else 'NOT_MEASURED'
    sizes=bool(packets) and all(len(r['packet_bytes'])==16+2*r['M'] for r in packets)
    exact_quant=bool(stages.loc['q_int16','exact_equal']) if quant_complete else None
    return {'encoder_p95':{'status':timing_status,'target_ms':100.,'samples_by_M':dict(zip(timing_table['M'].astype(str),map(int,counts)))},
            'additional_ram':{'status':('PASS' if 0<=ram<65536 else 'FAIL') if ram_valid else 'NOT_MEASURED',
                              'target_bytes':65536,'value_bytes':float(ram) if ram_valid else np.nan},
            'quantization':{'status':quant_status,'bit_exact':exact_quant,'max_lsb_diff':stages.loc['q_int16','max_lsb_diff']},
            'packet_crc':{'status':packet_status,'received_packets':len(packets)},
            'packet_size':{'status':'PASS' if packet_complete and sizes else 'PARTIAL' if packets and sizes else 'FAIL' if packets else 'NOT_MEASURED'},
            'CRC_implementation':{'status':result.get('crc_common_vector_status','NOT_MEASURED')},
            'unknown_config_rejection':{'status':stage_status('unknown config_id rejection')}}


def resolve_eda_hardware_port(config,registry,ports):
    """Use configured port, or a unique USB board with matching physical INFO."""
    if not config.hardware_port:return None
    if config.hardware_port in [p.device for p in ports]:return config.hardware_port
    if not registry:return None
    matches=[]
    for port in ports:
        if getattr(port,'vid',None) is None:continue
        try:
            with _eda_connect(port.device) as connection:info=_eda_query_info(connection)
            if (info.get('FIRMWARE_VERSION')=='eda-raw-stream-crc-v1' and
                info.get('PROTOCOL_SHA')==_eda_contract_hash(config.numerical_contract()) and
                info.get('REGISTRY_SHA')==_eda_contract_hash({int(k):v for k,v in registry.items()})):
                matches.append(port.device)
        except (OSError,ValueError,TimeoutError):continue
    if len(matches)>1:raise ValueError('Multiple matching ESP32 boards; configure a specific hardware_port')
    return matches[0] if matches else None


def run_eda_hardware(config, raw_bvp_dataset, frames=1000, measure_memory=True,decoder_callback=None):
    """Run actual replay. measure_memory=True builds/flashes baseline then restores.

    NOT_MEASURED means no local serial port. PARTIAL means missing measurement or
    interrupted collection. MEASURED means complete raw replay/benchmark; exact
    packet equivalence remains an independent stage and may legitimately fail.
    """
    if not isinstance(frames,int) or frames<1:raise ValueError('frames must be a positive integer')
    out=config.directory('results');session_id=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]
    logdir=out/'hardware'/session_id;logdir.mkdir(parents=True,exist_ok=False)
    timings=[];stages=[];packets=[];ram=None;info={};audits=[];receiver_tests=pd.DataFrame()
    result={'status':'NOT_MEASURED','info':info,'packet_records':packets,'measurement_session_id':session_id,
            'reason':'Chưa có phép đo thực tế','measured_frames':0,'requested_frames_per_M':frames,
            'warmup_per_M':32,'crc_common_vector_status':'NOT_MEASURED',
            'firmware_source_sha256':_eda_file_sha256(Path(config.root)/'esp32_firmware/src/main.cpp') if (Path(config.root)/'esp32_firmware/src/main.cpp').exists() else None,
            'collection_complete':False,'ram_measurement_complete':False}
    # Reset current-run evidence before discovery; never load a previous result.
    _eda_save_json(out/'hardware_status.json',{'status':'running','measurement_session_id':session_id})
    _eda_save_json(out/'ram_comparison.json',{'status':'NOT_MEASURED','measurement_session_id':session_id})
    try:
        from serial.tools import list_ports
        ports=list(list_ports.comports())
        registry_path=config.directory('exports')/'config_registry.json'
        discovery_registry=json.loads(registry_path.read_text(encoding='utf-8')) if registry_path.exists() else None
        actual_port=resolve_eda_hardware_port(config,discovery_registry,ports)
        result['configured_hardware_port']=config.hardware_port
        result['hardware_port']=actual_port
        if actual_port and actual_port!=config.hardware_port:
            from dataclasses import replace
            config=replace(config,hardware_port=actual_port)
        devices=[p.device for p in ports]
        if not config.hardware_port or config.hardware_port not in devices:
            result['reason']='Local serial port unavailable; cần chạy ESP32 trên PC/local runtime'
        else:
            registry=json.loads((config.directory('exports')/'config_registry.json').read_text(encoding='utf-8'))
            _eda_validate_registry(config,registry)
            count=576+128*(frames+32-1);raw_by_subject={}
            for fold in range(1,6):
                subject=f'S{(fold-1)*3+1}'
                raw,audit=audit_eda_raw(raw_bvp_dataset,subject,count if fold==1 else 576+7*128,config.data_root)
                raw_by_subject[subject]=raw;audits.append(audit)
            _eda_save_json(logdir/'raw_input_audit.json',audits)
            with _eda_connect(config.hardware_port) as port:
                serial_port=_EDASerialCapture(port,logdir)
                before=_eda_query_info(serial_port);info.update(before=before)
                if before.get('PROTOCOL_SHA')!=_eda_contract_hash(config.numerical_contract()) or before.get('REGISTRY_SHA')!=_eda_contract_hash({int(k):v for k,v in registry.items()}):
                    raise ValueError('Hardware protocol/registry fingerprint mismatch')
                crc_rows=[verify_eda_crc_vector(serial_port,value) for value in (b'123456789',bytes(range(64)))]
                stages.extend(crc_rows)
                result['crc_common_vector_status']='FAIL' if any(r['status']=='FAIL' for r in crc_rows) else 'PASS' if all(r['status']=='PASS' for r in crc_rows) else 'NOT_MEASURED'
                if result['crc_common_vector_status']=='FAIL':raise ValueError('CRC implementation common-vector mismatch')
                serial_port.write(b'STREAM_BEGIN 9999 576 0\n')
                rejected=serial_port.readline().strip()==b'ERR_CONFIG'
                stages.append({'stage':'unknown config_id rejection','status':'PASS' if rejected else 'FAIL',
                               'exact_equal':rejected,'detail':'Expected=REJECT; Actual='+('REJECT' if rejected else 'ACCEPT/INVALID RESPONSE')})
                if not rejected:raise ValueError('Unknown config_id not rejected by firmware')
                stages.extend({'stage':'raw input identity','status':'PASS','exact_equal':True,
                               'detail':'Full array equals original wrist/BVP; sample 0 and transient retained'} for _ in audits)
                for fold in range(1,6):
                    subject=f'S{(fold-1)*3+1}'
                    for M in config.M_list:
                        _eda_collect_session(serial_port,registry,fold*1000+M,subject,raw_by_subject[subject],8,True,0,session_id,timings,stages,packets,logdir,decoder_callback)
                for M in config.M_list:
                    _eda_collect_session(serial_port,registry,1000+M,'S1',raw_by_subject['S1'],frames+32,False,32,session_id,timings,stages,packets,logdir,decoder_callback)
                receiver_tests=test_eda_packet_receiver(packets[0]['packet_bytes'],registry)
                unknown=receiver_tests.set_index('test').loc['unknown config_id']
                stages.append({'stage':'unknown config_id rejection','status':unknown['status'],
                               'exact_equal':unknown['actual']=='REJECT',
                               'detail':'Receiver: CRC-valid unknown config packet; Expected=REJECT; Actual='+unknown['actual']})
                after=_eda_query_info(serial_port);info.update(after=after)
            numeric_failed=any(r['status']=='FAIL' for r in stages if r['stage'] not in ('payload bytes','packet bytes'))
            result['collection_complete']=True
            result['status']='FAILED' if numeric_failed else 'MEASURED'
            result['reason']='Stage numerical/receiver check failed' if numeric_failed else 'Actual raw-stream HIL measured'
            if not numeric_failed and result['crc_common_vector_status']!='PASS':
                result['status']='PARTIAL';result['reason']='Raw-stream measured; common-vector CRC firmware command unavailable'
            full_status={'status':'failed' if numeric_failed else 'measured','measurement_session_id':session_id,'hardware_port':config.hardware_port,
                         'info_before':before,'info_after':after,'warmup_per_M':32}
            _eda_save_json(out/'hardware_status.json',full_status)
            if measure_memory and not numeric_failed:
                if frames!=1000:
                    result['status']='PARTIAL';result['reason']='Matched RAM helper requires exactly 1000 benchmark frames per M'
                else:
                    from measure_ram import measure_ram as _eda_measure_ram
                    capture=io.StringIO()
                    with contextlib.redirect_stdout(capture),contextlib.redirect_stderr(capture):
                        ram=_eda_measure_ram(config)
                    (logdir/'ram_helper_output.log').write_text(capture.getvalue(),encoding='utf-8')
                    result['ram_measurement_complete']=ram.get('status')=='measured' and ram.get('measurement_session_id')==session_id
            result['raw_input_audit']=audits
    except ImportError:
        result['status']='NOT_MEASURED' if not packets else 'PARTIAL'
        result['reason']='Required local hardware dependency unavailable'
    except Exception as exc:
        (logdir/'error.log').write_text(f'{type(exc).__name__}: {exc}',encoding='utf-8')
        result['status']='PARTIAL' if packets else 'FAILED'
        result['reason']=f'Hardware run incomplete ({type(exc).__name__}); inspect saved diagnostic log'
    result['measured_frames']=sum(not r['trace'] and not r['warmup'] for r in timings)
    result['benchmark_frames_by_M']={str(M):sum(r['M']==M and not r['trace'] and not r['warmup'] for r in timings) for M in config.M_list}
    result['stage_table']=_eda_stage_table(stages)
    result['timing_table']=summarize_eda_timing(timings,config.M_list)
    result['ram_table']=summarize_eda_ram(info.get('after',{}),ram)
    result['receiver_test_table']=receiver_tests
    result['acceptance']=derive_eda_hardware_acceptance(result,result['stage_table'],result['timing_table'],result['ram_table'],packets,frames)
    result['timing_records']=timings;result['stage_records']=stages;result['ram_comparison']=ram
    result['timing_note']='Device encoder_us starts after normalization and ends after packet/CRC; excludes UART, filtering and normalization. Trace and 32 warmup frames/M excluded.'
    pd.DataFrame(timings).to_csv(out/'hardware_timings.csv',index=False)
    pd.DataFrame(stages).to_csv(out/'hardware_traces.csv',index=False)
    receiver_tests.to_csv(logdir/'receiver_rejection_tests.csv',index=False)
    status={k:v for k,v in result.items() if k not in ('packet_records','stage_table','timing_table','ram_table','timing_records','stage_records','receiver_test_table')}
    _eda_save_json(out/'eda_hardware_status.json',status)
    if result['status']!='MEASURED':
        _eda_save_json(out/'hardware_status.json',{'status':result['status'].lower(),'measurement_session_id':session_id,
                                                'measured_frames':result['measured_frames'],'reason':result['reason']})
    return result
