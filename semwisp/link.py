"""RFID link model: the uplink payload travels as EPC words, and a word that fails its CRC is lost whole."""
from dataclasses import dataclass

import numpy as np


@dataclass
class Received:
    data: np.ndarray      # uint8 payload; bytes of lost EPCs are 0
    ok: np.ndarray        # bool per byte: False where the EPC was lost


class Link:
    def __init__(self, epc_bytes=12, p_loss=0.0, seed=0):
        """epc_bytes: bytes per EPC (12 = the common 96-bit EPC). p_loss: independent loss probability per EPC."""
        self.epc_bytes, self.p_loss = epc_bytes, p_loss
        self.rng = np.random.default_rng(seed)

    def n_epcs(self, n):
        """(EPC count, payload bytes per EPC) for an n-byte payload. A payload that spans several EPCs spends
        the first bytes of each on a chunk index, so the server can order them and see which are missing."""
        if n <= self.epc_bytes:
            return 1, self.epc_bytes
        idx = 1
        while True:
            per = self.epc_bytes - idx
            k = -(-n // per)
            if k <= 256 ** idx:
                return k, per
            idx += 1

    def uplink(self, payload, ledger):
        payload = np.asarray(payload, np.uint8)
        k, per = self.n_epcs(len(payload))
        ledger.up_bytes += len(payload)
        ledger.up_epcs += k
        ok = np.ones(len(payload), bool)
        if self.p_loss:
            ok = np.repeat(self.rng.random(k) >= self.p_loss, per)[:len(payload)]
        return Received(np.where(ok, payload, 0).astype(np.uint8), ok)
