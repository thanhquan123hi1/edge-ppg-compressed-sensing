"""
quantization.py - Symmetric INT16 Quantization, Packet Framing, and CRC32.
Adheres strictly to proposal Section 2.4, 2.5 and Table 3:
- Scale delta = max(max|y| / 32767, 1e-12) as Float32. (If y==0, delta=1.0).
- Round-half-away-from-zero: roundf equivalent across Python and firmware.
- Symmetric INT16 clip [-32767, 32767], level -32768 not used.
- Dequantized measurement: y_tilde = q * delta.
- 16-byte header:
  version (uint8, 1B), flags (uint8, 1B), config_id (uint16, 2B),
  frame_index (uint32, 4B), scale delta (float32, 4B), crc32 (uint32, 4B).
  CRC32 covers the first 12 bytes of header + entire payload.
"""

import struct
import zlib
import numpy as np

HEADER_FORMAT_PART1 = "<BBHIf"  # 12 bytes: version, flags, config_id, frame_index, scale_delta
HEADER_SIZE = 16
INT16_MAX = 32767
EPSILON_SCALE = 1e-12

def round_half_away_from_zero(val):
    """
    Symmetric rounding matching standard C roundf():
    Halfway cases are rounded away from zero.
    """
    x = np.asarray(val,dtype=np.float64)
    return np.sign(x) * np.floor(np.abs(x) + 0.5)

def quantize_int16(y):
    """
    Symmetric INT16 quantization of measurement vector y.
    y: 1D array of shape [M] or 2D array of shape [B, M]
    Returns:
      q: int16 array of same shape
      delta: float32 scale factor (scalar or shape [B, 1])
      y_tilde: dequantized float32 array
      clip_count: number of elements clipped
    """
    y = np.asarray(y, dtype=np.float32)
    if y.ndim not in (1,2) or not np.isfinite(y).all() or y.shape[-1] == 0:
        raise ValueError('Measurements must be a nonempty finite vector or matrix')
    is_1d = (y.ndim == 1)
    if is_1d:
        y = y[np.newaxis, :]
        
    a = np.max(np.abs(y), axis=1, keepdims=True)
    exact_scale = np.maximum(a.astype(np.float64)/INT16_MAX,EPSILON_SCALE)
    delta = exact_scale.astype(np.float32)
    below = delta.astype(np.float64) < exact_scale
    delta[below] = np.nextafter(delta[below],np.float32(np.inf))
    # If all zeros in a row, set delta = 1.0
    zero_rows = (a == 0.0)
    delta[zero_rows] = 1.0
    
    scaled = y / delta
    rounded = round_half_away_from_zero(scaled)
    
    clip_mask = (rounded < -INT16_MAX) | (rounded > INT16_MAX)
    clip_count = int(np.sum(clip_mask))
    
    q = np.clip(rounded, -INT16_MAX, INT16_MAX).astype(np.int16)
    with np.errstate(over='ignore'):
        y_tilde = (q.astype(np.float32) * delta).astype(np.float32)
    if not np.isfinite(y_tilde).all():raise ValueError('Float32 dequantization overflow; finite reconstruction unrepresentable')
    
    if is_1d:
        return q[0], float(delta[0, 0]), y_tilde[0], clip_count
    return q, delta, y_tilde, clip_count

def pack_packet(q_int16, delta, frame_index, config_id, version=1, flags=0):
    """
    Packs quantized INT16 measurements into a 16-byte header + 2M payload packet.
    """
    q_bytes = q_int16.astype("<i2").tobytes()
    header_12 = struct.pack(HEADER_FORMAT_PART1, version, flags, config_id, frame_index, float(delta))
    crc = zlib.crc32(header_12 + q_bytes) & 0xffffffff
    full_header = header_12 + struct.pack("<I", crc)
    return full_header + q_bytes

def unpack_packet(packet_bytes,config_registry=None):
    """
    Unpacks packet, verifies CRC32, returns fields and dequantized y_tilde.
    """
    if len(packet_bytes) < HEADER_SIZE:
        raise ValueError(f"Packet too short: {len(packet_bytes)} < {HEADER_SIZE}")
        
    header_12 = packet_bytes[:12]
    expected_crc = struct.unpack("<I", packet_bytes[12:16])[0]
    payload = packet_bytes[16:]
    
    actual_crc = zlib.crc32(header_12 + payload) & 0xffffffff
    if actual_crc != expected_crc:
        raise ValueError(f"CRC32 mismatch! Expected 0x{expected_crc:08X}, got 0x{actual_crc:08X}")
        
    version, flags, config_id, frame_index, delta = struct.unpack(HEADER_FORMAT_PART1, header_12)
    if version != 1:
        raise ValueError('Unsupported version')
    if flags not in (0,1):
        raise ValueError('Unsupported flags')
    if config_registry is None or (config_id not in config_registry and str(config_id) not in config_registry):
        raise ValueError(f'Unknown config_id {config_id}')
    cfg = config_registry.get(config_id,config_registry.get(str(config_id)))
    expected_bytes = int(cfg['M']) * (2 if flags == 0 else 4)
    if len(payload) != expected_bytes:
        raise ValueError(f'Wrong payload length {len(payload)} != {expected_bytes}')
    if not np.isfinite(delta) or delta <= 0:
        raise ValueError('Scale must be positive finite')
    
    if flags == 0:  # INT16 mode
        q = np.frombuffer(payload, dtype="<i2")
        if np.any(q == -32768):
            raise ValueError('Reserved INT16 value -32768')
        with np.errstate(over='ignore'):
            y_tilde = q.astype(np.float32) * delta
        if not np.isfinite(y_tilde).all():raise ValueError('Packet dequantization overflow/nonfinite input')
    else:  # Float32 mode
        y_tilde = np.frombuffer(payload, dtype="<f4")
        q = None
        if delta != 1 or not np.isfinite(y_tilde).all():
            raise ValueError('Invalid Float32 payload/scale')
        
    return {
        "version": version,
        "flags": flags,
        "config_id": config_id,
        "frame_index": frame_index,
        "delta": delta,
        "q": q,
        "y_tilde": y_tilde,
        "crc32": expected_crc
    }

if __name__ == "__main__":
    print("Testing quantization and packet framing...")
    M = 77
    y = np.random.randn(M).astype(np.float32) * 5.0
    q, delta, y_tilde, clip_cnt = quantize_int16(y)
    packet = pack_packet(q, delta, frame_index=42, config_id=77)
    print(f"Packet size: {len(packet)} bytes (Expected: {2*M + 16} bytes)")
    assert len(packet) == 2*M + 16, f"Size mismatch: {len(packet)} != {2*M + 16}"
    
    unpacked = unpack_packet(packet,{77:{'M':77}})
    print("CRC32 verified successfully: 0x{:08X}".format(unpacked["crc32"]))
    max_recon_diff = np.max(np.abs(y_tilde - unpacked["y_tilde"]))
    print(f"Max dequantization unpacking diff: {max_recon_diff:.2e}")
    assert max_recon_diff == 0.0, "Dequantization diff is non-zero!"
    print("Quantization & Framing module verified successfully.")
