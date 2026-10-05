"""
preprocess.py - Causal SOS filtering, normalization, and windowing routines.
Adheres strictly to proposal Section 2.3:
- Butterworth bandpass filter order 4 (2 second-order sections): scipy.signal.butter(2, [0.5, 8.0], btype="bandpass", fs=64, output="sos")
- Must run via sosfilt with state preserved across chunks/samples.
- DO NOT use filtfilt (non-causal forward-backward filter).
- Drops first 5s (320 samples) after initialization to eliminate transient.
"""

import numpy as np
import scipy.signal

FS = 64
LOW_CUT = 0.5
HIGH_CUT = 8.0
FILTER_ORDER = 2  # 2nd order lowpass/highpass prototype -> 4th order bandpass (2 SOS sections)
DROP_SECONDS = 5
DROP_SAMPLES = DROP_SECONDS * FS  # 320 samples

def design_causal_sos(fs=FS, low_cut=LOW_CUT, high_cut=HIGH_CUT, order=FILTER_ORDER):
    """Designs causal Butterworth bandpass SOS filter."""
    sos = scipy.signal.butter(order, [low_cut, high_cut], btype="bandpass", fs=fs, output="sos")
    return sos

class CausalSOSFilter:
    """
    Streaming causal SOS filter with preserved state.
    Matches ESP32 Direct Form II Transposed implementation.
    """
    def __init__(self, sos=None):
        if sos is None:
            sos = design_causal_sos()
        self.sos = sos.astype(np.float32)
        self.n_sections = self.sos.shape[0]
        self.reset()
        
    def reset(self):
        # Initial condition zi = 0 as specified in proposal
        self.zi = np.zeros((self.n_sections, 2), dtype=np.float32)
        
    def filter_chunk(self, x):
        """Filters a continuous chunk of 1D signal x, updating state in-place."""
        x = np.asarray(x, dtype=np.float32)
        y, self.zi = scipy.signal.sosfilt(self.sos, x, zi=self.zi)
        return y.astype(np.float32)
        
    def filter_sample(self, sample):
        """Processes a single sample through the 2 SOS sections."""
        out = sample
        for s in range(self.n_sections):
            b0, b1, b2, a0, a1, a2 = self.sos[s]
            # Direct Form II Transposed:
            # y[n] = b0*x[n] + d0
            # d0 = b1*x[n] - a1*y[n] + d1
            # d1 = b2*x[n] - a2*y[n]
            w = b0 * out + self.zi[s, 0]
            self.zi[s, 0] = b1 * out - a1 * w + self.zi[s, 1]
            self.zi[s, 1] = b2 * out - a2 * w
            out = w
        return float(out)

def normalize_signal(signal, mu, sigma):
    """Z-score normalization using fixed reference parameters."""
    return (signal - mu) / sigma

def denormalize_signal(signal_norm, mu, sigma):
    """Inverts z-score normalization."""
    return signal_norm * sigma + mu

if __name__ == "__main__":
    print("Testing CausalSOSFilter...")
    sos = design_causal_sos()
    filt = CausalSOSFilter(sos)
    x = np.random.randn(1000).astype(np.float32)
    y_chunk = filt.filter_chunk(x)
    
    filt.reset()
    y_sample = np.array([filt.filter_sample(val) for val in x], dtype=np.float32)
    max_diff = np.max(np.abs(y_chunk - y_sample))
    print(f"Max diff between chunk and sample filtering: {max_diff:.2e}")
    assert max_diff < 1e-5, "Mismatch between chunk and sample filtering!"
    print("CausalSOSFilter verified successfully.")
