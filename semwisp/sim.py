"""Replay loop: one query per frame, in time order, between a simulated tag and a server.

Per query the server sends requests and the tag answers each with a payload, until the server stops asking:

    server.begin(t)
    while (req := server.request(round)) is not None:
        payload = tag.respond(req)                 # bytes, exactly what would be backscattered
        server.receive(req, link.uplink(payload))
    score = server.decide()                        # higher = more likely a person

A fixed encoder is one round with a constant request. A progressive scheme is several rounds. An adaptive
scheme sends requests that change the tag's stored configuration. Everything a query costs is counted in a
Ledger, so all methods are compared on the same axes.
"""
from dataclasses import dataclass, field, fields

import pandas as pd

from .link import Link
from .sensor import Sensor


@dataclass
class Ledger:
    """What one query cost. Counts only; cost.py turns them into energy."""
    rounds: int = 0
    down_bits: int = 0      # request bits beyond a plain inventory round
    up_bytes: int = 0       # payload bytes
    up_epcs: int = 0        # EPC words backscattered
    pixels_read: int = 0    # pixels clocked out of the sensor
    mcu_ops: int = 0        # rough count of add/shift/compare/store operations on the tag
    ram_bytes: int = 0      # peak working memory on the tag, including the payload buffer


@dataclass
class Request:
    op: str
    args: dict = field(default_factory=dict)
    n_bits: int = 0         # downlink size of this request


class Tag:
    """Tag-side program. Subclasses implement respond(); state set in reset() persists across queries."""

    def reset(self):
        """Power-on state, called at the start of each sequence."""

    def new_query(self, sensor, scene, ledger):
        self._sensor, self._scene, self._frame, self.ledger = sensor, scene, None, ledger

    def frame(self):
        """The current frame, captured on first use and then held for the rest of the query."""
        if self._frame is None:
            self._frame = self._sensor.capture(self._scene, self.ledger)
        return self._frame

    def respond(self, request):
        raise NotImplementedError


class Server:
    def reset(self):
        """Called at the start of each sequence."""

    def begin(self, t):
        pass

    def request(self, round):
        raise NotImplementedError

    def receive(self, request, rx):
        raise NotImplementedError

    def decide(self):
        raise NotImplementedError


@dataclass
class Trace:
    df: pd.DataFrame        # one row per query: seq, idx, t, label, score and the Ledger fields
    payloads: list          # per query, the list of Received objects (empty unless keep_payloads)


def replay(seqs, tag, server, link=None, sensor=None, stride=1, max_rounds=16, keep_payloads=False):
    """Run every `stride`-th non-corrupted frame of each sequence in `seqs` (from frames.load_frames)."""
    link, sensor = link or Link(), sensor or Sensor()
    rows, payloads = [], []
    for name, s in seqs.items():
        tag.reset()
        server.reset()
        for i in range(0, len(s['X']), stride):
            if s['bad'][i]:
                continue
            ledger, got = Ledger(), []
            tag.new_query(sensor, s['X'][i], ledger)
            server.begin(s['t'][i])
            while ledger.rounds < max_rounds and (req := server.request(ledger.rounds)) is not None:
                ledger.rounds += 1
                ledger.down_bits += req.n_bits
                rx = link.uplink(tag.respond(req), ledger)
                server.receive(req, rx)
                if keep_payloads:
                    got.append(rx)
            rows.append(dict(seq=name, idx=i, t=s['t'][i], label=int(s['label'][i]) if 'label' in s else -1,
                             score=server.decide(), **{f.name: getattr(ledger, f.name) for f in fields(ledger)}))
            payloads.append(got)
    return Trace(pd.DataFrame(rows), payloads)
