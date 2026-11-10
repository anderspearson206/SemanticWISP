"""Ledger counts -> energy per query on the tag.

Energy, not bytes, is the real budget of a battery-free tag, but most of the constants needed here are NOT
known yet. Each one records where its value came from; a constant that is None has to be measured on the
hardware (or found in a datasheet) before its component can be computed, and until then that component is NaN.
Do not quote energy numbers from this module without checking the constants first.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class Const:
    value: float | None     # joules per unit
    per: str
    basis: str


MSP430_WISP = dict(
    capture=Const(None, 'frame',
                  'HM01B0 draws 1.1 mW in QQVGA mode (WISPCam-style board description), so this is 1.1 mW x the '
                  'time the sensor must stay on per frame, including wake-up and exposure settling. Measure.'),
    mcu_op=Const(None, 'counted operation',
                 'MSP430FR5969 datasheet headline is about 100 uA/MHz in active mode; needs the supply voltage and '
                 'the clock cycles per counted operation (several, more for FRAM writes). Measure on a real loop.'),
    epc=Const(None, 'EPC backscattered', 'Tag energy for one Gen2 inventory round with a 96-bit reply. Measure.'),
    down_bit=Const(None, 'downlink bit', 'Tag energy to receive and decode one extra reader-to-tag bit. Measure.'),
)

COUNT_OF = dict(capture=lambda df: df.pixels_read / 19200, mcu_op=lambda df: df.mcu_ops,
                epc=lambda df: df.up_epcs, down_bit=lambda df: df.down_bits)


def energy(df, platform=MSP430_WISP):
    """Adds one column per component (J per query, NaN where the constant is unknown) and their total `energy_j`
    to a copy of a Trace dataframe."""
    out = df.copy()
    for name, c in platform.items():
        out[f'e_{name}'] = COUNT_OF[name](df) * (np.nan if c.value is None else c.value)
    out['energy_j'] = out[[f'e_{n}' for n in platform]].sum(axis=1, skipna=False)
    return out
