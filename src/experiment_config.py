"""Locked versioned experiment settings shared by notebook and CLI."""
from dataclasses import dataclass, asdict
from pathlib import Path
import scipy.signal
import numpy as np

@dataclass(frozen=True)
class ExperimentConfig:
    root: str = 'D:/IOT/CuoiKy'
    data_root: str = 'E:/STUDY/DATASET/PPG_FieldStudy'
    run_id: str = '20261004_three_models_v2'
    fs: int = 64
    N: int = 256
    H: int = 128
    drop: int = 320
    matrix_seed: int = 42
    seeds: tuple = (11, 22, 33)
    M_list: tuple = (51, 77, 128)
    max_epochs: int = 100
    patience: int = 10
    batch_size: int = 64
    learning_rate: float = .001
    protocol_version: str = 'cs-int16-upward-half-away-v2'
    hardware_port: str = 'COM6'
    hardware_required: bool = True

    def __post_init__(self):
        if (self.fs,self.N,self.H,self.drop,self.matrix_seed)!=(64,256,128,320,42) or tuple(self.M_list)!=(51,77,128) or tuple(self.seeds)!=(11,22,33):
            raise ValueError('Numerical experiment contract is locked; change the design before changing Fs/N/H/drop/matrix/seeds')
        if self.max_epochs<1 or self.patience<1 or self.batch_size<1 or self.learning_rate<=0:
            raise ValueError('Invalid training budget')

    def directory(self, kind):
        p = Path(self.root) / kind / self.run_id
        p.mkdir(parents=True, exist_ok=True)
        return p

    def numerical_contract(self):
        return {'fs':self.fs,'N':self.N,'H':self.H,'drop':self.drop,
                'matrix_seed':self.matrix_seed,'protocol':self.protocol_version,
                'filter_dtype':'float32','sos':filter_sos().tolist(),
                'normalization':'train-samples-float64-statistics-float32-application'}

def filter_sos():
    return scipy.signal.butter(2,[.5,8.],btype='bandpass',fs=64,output='sos').astype(np.float32)
