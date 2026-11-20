import torch


class JITFunction:
    def __init__(self, fn):
        self.fn = fn

    def __getitem__(self, grid):
        return lambda *args, **kwargs: self.run(*args, grid=grid, warmup=False, **kwargs)

    def run(self, *args, grid, warmup, **kwargs):
        views = [a.detach().numpy() if isinstance(a, torch.Tensor) else a for a in args]
        return self.fn(*views, **kwargs)
