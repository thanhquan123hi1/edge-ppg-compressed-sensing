"""Build/upload comparable firmware, measure actual baseline, restore full encoder."""
import json
import re
import subprocess
from pathlib import Path
from dataset import load_raw_bvp
from verify_esp32 import connect,query_info
from provenance import save_json,file_sha256

def firmware_tools():
    base=Path.home()/'.platformio';return base/'penv/Scripts/platformio.exe',base/'packages/toolchain-xtensa-esp32s3/bin/xtensa-esp32s3-elf-nm.exe',base/'packages/toolchain-xtensa-esp32s3/bin/xtensa-esp32s3-elf-size.exe'

def build_upload(config,project,upload=True):
    pio,_,_=firmware_tools();directory=config.directory('results');cmd=[str(pio),'run','-d',str(Path(config.root)/project)]
    if upload:cmd+=['-t','upload','--upload-port',config.hardware_port]
    proc=subprocess.run(cmd,capture_output=True,text=True,encoding='utf-8',errors='replace')
    (directory/(project+('_upload.log' if upload else '_build.log'))).write_text(proc.stdout+proc.stderr,encoding='utf-8')
    if proc.returncode:raise RuntimeError(f'Firmware {project} build/upload failed; see log')
    return proc.stdout

def elf_evidence(config,project):
    _,nm,size=firmware_tools();elf=Path(config.root)/project/'.pio/build/esp32s3/firmware.elf'
    symbols=subprocess.check_output([str(nm),'-S','--size-sort',str(elf)],text=True)
    sections=subprocess.check_output([str(size),'-A',str(elf)],text=True)
    dest=config.directory('results');(dest/(project+'_symbols.txt')).write_text(symbols);(dest/(project+'_sections.txt')).write_text(sections)
    section_sizes={line.split()[0]:int(line.split()[1]) for line in sections.splitlines() if re.match(r'^\.[^\s]+\s+\d+\s+',line)}
    static=sum(v for k,v in section_sizes.items() if k in ['.dram0.data','.dram0.bss','.noinit'])
    return {'elf_sha256':file_sha256(elf),'sections':section_sizes,'static_dram_bytes':static,
            'matrix_symbols':[line for line in symbols.splitlines() if 'B_PACKED' in line]}

def measure_ram(config):
    out=config.directory('results');full_status=json.loads((out/'hardware_status.json').read_text(encoding='utf-8'))
    if full_status['status']!='measured':raise RuntimeError('Full encoder must be physically measured before RAM comparison')
    full_info=full_status['info_after'];full_elf=elf_evidence(config,'esp32_firmware');raw=load_raw_bvp('S1',config.data_root)
    try:
        build_upload(config,'esp32_replay_baseline')
        with connect(config.hardware_port) as ser:
            baseline_before=query_info(ser);total_frames=1000+int(full_status.get('warmup_per_M',0));count=576+128*(total_frames-1);ser.write(f'REPLAY {count}\n'.encode())
            if ser.readline().strip()!=b'READY_BASELINE':raise RuntimeError('Baseline replay not ready')
            for frame in range(total_frames):
                lo,hi=(0,576) if frame==0 else (576+128*(frame-1),576+128*frame)
                ser.write(raw[lo:hi].astype('<f4').tobytes())
                if ser.readline().strip()!=f'ACK {hi}'.encode():raise RuntimeError('Baseline replay cadence failure')
            if ser.readline().strip()!=b'BASELINE_DONE':raise RuntimeError('Baseline replay incomplete')
            baseline_after=query_info(ser)
        baseline_elf=elf_evidence(config,'esp32_replay_baseline')
    finally:
        build_upload(config,'esp32_firmware')
    # Same runtime reservation; static already reduces available heap. Heap-free
    # difference therefore covers static plus extra allocations; do not add twice.
    delta_peak=int(baseline_after['INTERNAL_MIN_FREE'])-int(full_info['INTERNAL_MIN_FREE'])
    result={'status':'measured','baseline_before':baseline_before,'baseline_after':baseline_after,'full_after':full_info,
       'baseline_elf':baseline_elf,'full_elf':full_elf,'incremental_static_bytes':full_elf['static_dram_bytes']-baseline_elf['static_dram_bytes'],
       'incremental_peak_reserved_internal_bytes':delta_peak,'under_64KiB':0<=delta_peak<65536,
       'baseline_frames_including_warmup':total_frames,
       'measurement_session_id':full_status.get('measurement_session_id'),
       'note':'Internal minimum-free comparison after matched 1000 benchmark frames plus same warmup raw replay; stack allocated from heap is not added again. Full encoder also exercised trace sessions. PSRAM absent on this board.'}
    save_json(out/'ram_comparison.json',result);return result

if __name__=='__main__':
    from experiment_config import ExperimentConfig
    print(measure_ram(ExperimentConfig()))
