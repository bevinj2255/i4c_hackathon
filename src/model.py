"""Restoration network: denoise + x2 super-resolve in one pass.

Design decisions and why:

  ALL WORK AT LOW RESOLUTION. Every convolution runs on the 128x128 input; a single
  PixelShuffle at the end produces 256x256. Doing the body at high resolution would
  cost 4x the FLOPs for the same depth. KLA scores end-to-end throughput on an H100
  and warns that "unnecessarily large models may lose throughput", so the cheap
  arrangement and the scored metric point the same way. It is also what makes the
  model trainable on a 4GB GTX 1650.

  NO GLOBAL SKIP FROM THE INPUT. Standard EDSR adds an upsampled copy of the input
  to the output, which works when the input is clean and merely small. Ours is noisy
  -- that skip would pipe the speckle straight into the prediction, which is the one
  thing we are trying to remove. The long skip runs from the head features instead,
  so gradients still reach the early layers. The overfit check in train.py confirms
  this still converges.

  FULLY CONVOLUTIONAL. No hardcoded sizes, so the same weights restore 128->256 and
  256->512. All released data is x2, but the brief mentions 512x512 ground truth too.

  PLAIN RESIDUAL BLOCKS. Not attention, not a transformer. With one overnight run on
  a 4GB card there is no budget to tune something exotic, and a residual CNN is the
  option with the least that can go wrong. Residual scaling (0.1) keeps deep stacks
  stable, which is the one trick here that is not obvious.
"""
import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, ch, res_scale=0.1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1),
        )
        self.res_scale = res_scale

    def forward(self, x):
        return x + self.body(x) * self.res_scale


class RestoreNet(nn.Module):
    def __init__(self, channels=64, blocks=16, scale=2, res_scale=0.1):
        super().__init__()
        self.scale = scale
        self.head = nn.Conv2d(1, channels, 3, padding=1)
        self.body = nn.Sequential(
            *[ResBlock(channels, res_scale) for _ in range(blocks)],
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        self.upsample = nn.Sequential(
            nn.Conv2d(channels, channels * scale * scale, 3, padding=1),
            nn.PixelShuffle(scale),
            nn.ReLU(inplace=True),
        )
        self.tail = nn.Conv2d(channels, 1, 3, padding=1)

    def forward(self, x):
        # Centre the input. NOT clipped: values outside [0,1] carry information about
        # the speckle, and KLA calls that "a feature not a bug".
        x = x - 0.5
        f = self.head(x)
        f = f + self.body(f)          # long skip from head features, not from the input
        return self.tail(self.upsample(f)) + 0.5

    @torch.no_grad()
    def restore(self, x):
        """Inference entry point. Clamps to [0,1] -- do not bypass this.

        Ground truth is guaranteed to live in [0,1] and KLA explicitly does not clip
        or renormalise what we save, so anything outside the range is free error we
        are handing them. Clamping lives here, in the one method inference calls, so
        it cannot be forgotten at the call site.
        """
        return self.forward(x).clamp_(0.0, 1.0)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


def build(cfg):
    return RestoreNet(
        channels=cfg.get("channels", 64),
        blocks=cfg.get("blocks", 16),
        scale=cfg.get("scale", 2),
        res_scale=cfg.get("res_scale", 0.1),
    )


def _demo():
    """Shape, scale-agnosticism and clamping must all hold."""
    torch.manual_seed(0)
    m = RestoreNet(channels=16, blocks=2)

    for h, w in ((128, 128), (256, 256), (64, 96)):
        y = m(torch.randn(2, 1, h, w))
        assert y.shape == (2, 1, h * 2, w * 2), f"{y.shape} wrong for {h}x{w}"

    # restore() must clamp even when forward() does not.
    wild = torch.full((1, 1, 32, 32), 50.0)
    assert m(wild).max() > 1.0, "test is useless if forward already stayed in range"
    r = m.restore(wild)
    assert r.min() >= 0.0 and r.max() <= 1.0, "restore() failed to clamp"

    # A model that cannot be broken is not being tested: zero the weights and the
    # output must go flat.
    for p in m.parameters():
        nn.init.zeros_(p)
    flat = m(torch.randn(1, 1, 32, 32))
    assert flat.std() < 1e-6, "zeroed model still produces structure -- forward() is wrong"

    print(f"CHECK: model.py self-check passed "
          f"(x2 at three input sizes, restore() clamps, zeroed model goes flat)")
    print(f"CHECK: RestoreNet(64,16) has {RestoreNet(64, 16).n_params():,} parameters")


if __name__ == "__main__":
    _demo()
