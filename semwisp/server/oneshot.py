import numpy as np

from ..sim import Request, Server


class OneShotServer(Server):
    """One plain query per frame, then decide. `unpack` turns the received payload into the decoder's input.

    With decoder=None it only collects (score is NaN): replay with keep_payloads=True to gather training data.
    """

    def __init__(self, unpack, decoder=None):
        self.unpack, self.decoder = unpack, decoder

    def begin(self, t):
        self.x = None

    def request(self, round):
        return Request('send') if round == 0 else None

    def receive(self, request, rx):
        self.x = self.unpack(rx)

    def decide(self):
        return np.nan if self.decoder is None else self.decoder.score(self.x)
