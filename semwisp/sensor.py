"""Image sensor model (Himax HM01B0 in QQVGA mode: 160x120, 8 bits per pixel).

The dataset frames stand in for the scene. PIROPO was recorded with a far better camera than the HM01B0, so
this model exists to test how robust a method is to sensor noise. The noise parameters are NOT calibrated: they
are knobs until real HM01B0 frames are available to fit them.
"""
import numpy as np

from .piropo import H, W


class Sensor:
    def __init__(self, read_noise=0.0, shot_noise=0.0, seed=0):
        """read_noise: std in gray levels, signal-independent. shot_noise: std = shot_noise * sqrt(gray level)."""
        self.read_noise, self.shot_noise = read_noise, shot_noise
        self.rng = np.random.default_rng(seed)

    def capture(self, scene, ledger):
        """One full QQVGA frame. The sensor clocks out every pixel, so the capture cost is the whole frame
        whatever the tag keeps."""
        assert scene.shape == (H, W) and scene.dtype == np.uint8
        ledger.pixels_read += scene.size
        if not (self.read_noise or self.shot_noise):
            return scene
        x = scene.astype(np.float32)
        std = np.sqrt(self.read_noise ** 2 + self.shot_noise ** 2 * x)
        noisy = x + std * self.rng.standard_normal(x.shape, dtype=np.float32)
        return np.clip(np.rint(noisy), 0, 255).astype(np.uint8)
