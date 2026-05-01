from abc import ABC, abstractmethod

import torch


class NormalizerBase(ABC):
    def __init__(self, eps: float = 1e-8):
        self.eps = eps

    @abstractmethod
    def fit(self, X: torch.Tensor) -> None:
        pass

    @abstractmethod
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        pass


class LimitsNormalizer(NormalizerBase):
    def fit(self, X: torch.Tensor) -> None:
        self.X = X
        if X.ndim > 2:
            X_flat = X.reshape(-1, X.shape[-1])
        else:
            X_flat = X

        self.mins = X_flat.min(dim=0).values
        self.maxs = X_flat.max(dim=0).values

        self.constant_mask = (self.maxs - self.mins).abs() < self.eps
        self.range = self.maxs - self.mins
        self.range = torch.where(
            self.constant_mask, torch.ones_like(self.range), self.range
        )

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mins) / self.range
        x = 2 * x - 1 + self.constant_mask.float()
        return x

    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.clip(x, -1, 1)
        x = (x + 1 - self.constant_mask.float()) / 2
        x = x * self.range + self.mins
        return x


class TrivialNormalizer(NormalizerBase):
    def fit(self, X: torch.Tensor) -> None:
        pass

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return x.clone()

    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x.clone()


class GaussianNormalizer(NormalizerBase):
    def fit(self, X: torch.Tensor) -> None:
        self.X = X
        if X.ndim > 2:
            X_flat = X.reshape(-1, X.shape[-1])
        else:
            X_flat = X

        self.mean = X_flat.mean(dim=0)
        self.std = X_flat.std(dim=0)

        self.constant_mask = self.std.abs() < self.eps
        self.std = torch.where(self.constant_mask, torch.ones_like(self.std), self.std)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std + self.mean


def get_data_transforms():
    return {
        "TrivialNormalizer": TrivialNormalizer,
        "LimitsNormalizer": LimitsNormalizer,
        "GaussianNormalizer": GaussianNormalizer,
    }
