"""Bit packing for payloads: n values of `bits` bits each <-> bytes (MSB first, zero-padded to a whole byte)."""
import numpy as np


def pack(values, bits):
    v = np.asarray(values, np.uint8).ravel()
    return np.packbits((v[:, None] >> np.arange(bits - 1, -1, -1)) & 1)


def unpack(payload, bits, n):
    b = np.unpackbits(np.asarray(payload, np.uint8))[:n * bits].reshape(n, bits)
    return (b << np.arange(bits - 1, -1, -1)).sum(1).astype(np.uint8)


def n_bytes(n, bits):
    return -(-n * bits // 8)
