"""
sensing.py - Compressed Sensing measurement matrix generation and bit-packing.
Adheres strictly to proposal Section 2.4:
- Measurement matrix: Rademacher B in {-1, +1}^(MxN), seed 42.
- Phi = B / sqrt(M) for M in {51, 77, 128}, N = 256.
- Bit-packing: Each row of B (256 bits) packed into 32 bytes (uint8).
  Bit k in byte b: 1 represents +1, 0 represents -1.
- Flash export: Generates C header file for ESP32 firmware with zero-overhead flash storage.
"""

import numpy as np

DEFAULT_SEED = 42
N_DEFAULT = 256

def generate_rademacher_matrix(M, N=N_DEFAULT, seed=DEFAULT_SEED):
    """
    Generates Rademacher matrix B in {-1, +1}^(M x N).
    Phi = B / sqrt(M).
    """
    rng = np.random.RandomState(seed)
    B = rng.choice([-1.0, 1.0], size=(M, N)).astype(np.float32)
    Phi = B / np.sqrt(M, dtype=np.float32)
    return B, Phi

def pack_rademacher_matrix(B):
    """
    Packs Rademacher matrix B (M x N) where elements are -1 or +1 into bit-packed uint8 array (M x N/8).
    bit=1 -> +1, bit=0 -> -1.
    """
    M, N = B.shape
    if N % 8 != 0:
        raise ValueError(f"N must be divisible by 8, got {N}")
    n_bytes = N // 8
    packed = np.zeros((M, n_bytes), dtype=np.uint8)
    
    for i in range(M):
        for j in range(N):
            if B[i, j] > 0:
                byte_idx = j // 8
                bit_idx = j % 8
                packed[i, byte_idx] |= (1 << bit_idx)
    return packed

def project_measurements(x, Phi):
    """
    Linear sensing projection: y = Phi @ x.
    x: shape [..., N]
    Phi: shape [M, N]
    Returns: shape [..., M]
    """
    return np.matmul(x, Phi.T, dtype=np.float32)

def export_c_header(filepath="d:/IOT/CuoiKy/esp32_firmware/src/cs_matrices.h", seed=DEFAULT_SEED):
    """
    Exports bit-packed matrices for M in {51, 77, 128} and filter SOS into a C header for ESP32.
    """
    import os
    import scipy.signal
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    sos = scipy.signal.butter(2, [0.5, 8.0], btype="bandpass", fs=64, output="sos").astype(np.float32)
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("/* Auto-generated Compressed Sensing matrices and filter coefficients */\n")
        f.write("#ifndef CS_MATRICES_H\n#define CS_MATRICES_H\n\n")
        f.write("#include <stdint.h>\n\n")
        f.write(f"#define CS_N {N_DEFAULT}\n")
        f.write(f"#define CS_N_BYTES {N_DEFAULT // 8}\n")
        f.write(f"#define CS_SEED {seed}\n\n")
        
        # SOS coefficients
        f.write("/* 2nd-order Butterworth bandpass filter (2 SOS sections, order 4) */\n")
        f.write("const float SOS_COEFFS[2][6] = {\n")
        for s in range(2):
            b0, b1, b2, a0, a1, a2 = sos[s]
            f.write(f"    {{{b0:.9e}f, {b1:.9e}f, {b2:.9e}f, {a0:.9e}f, {a1:.9e}f, {a2:.9e}f}},\n")
        f.write("};\n\n")
        
        for M in [51, 77, 128]:
            B, _ = generate_rademacher_matrix(M, N=N_DEFAULT, seed=seed)
            packed = pack_rademacher_matrix(B)
            inv_sqrt_M = 1.0 / np.sqrt(M)
            f.write(f"/* Measurement matrix B packed for M={M}, size={M}x{N_DEFAULT//8} bytes */\n")
            f.write(f"#define CS_INV_SQRT_M_{M} {inv_sqrt_M:.9e}f\n")
            f.write(f"const uint8_t B_PACKED_{M}[{M}][{N_DEFAULT//8}] = {{\n")
            for i in range(M):
                f.write("    {")
                f.write(", ".join([f"0x{val:02X}" for val in packed[i]]))
                f.write("},\n")
            f.write("};\n\n")
            
        f.write("#endif // CS_MATRICES_H\n")
    print(f"Exported C header to {filepath}")

if __name__ == "__main__":
    print("Testing sensing matrix generation and packing...")
    for M in [51, 77, 128]:
        B, Phi = generate_rademacher_matrix(M, N=256, seed=42)
        packed = pack_rademacher_matrix(B)
        print(f"M={M}: B shape={B.shape}, packed shape={packed.shape} ({packed.nbytes} bytes)")
    export_c_header()
