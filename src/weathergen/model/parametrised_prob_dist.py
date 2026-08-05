# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import numpy as np
import torch
import torch.nn as nn

from weathergen.model.norms import SaturateEncodings


class DiagonalGaussianDistribution:
    """
    Used to represent a learned Gaussian Distribution as typical in a VAE
    Code taken and adapted from: https://github.com/Jiawei-Yang/DeTok/tree/main
    """

    def __init__(self, deterministic=False, channel_dim=1):
        self.deterministic = deterministic
        self.channel_dim = channel_dim
        self.parameters = None
        self.mean = None
        self.logvar = None
        self.sum_dims = None
        self.std = None
        self.var = None

    def reset_parameters(self, parameters):
        self.parameters = parameters.float()
        # chunk the float-cast parameters: under autocast the incoming tensor is bf16 and
        # logvar.exp() overflows well before the -30/20 clamp range is exhausted
        self.mean, self.logvar = torch.chunk(self.parameters, 2, dim=self.channel_dim)
        self.sum_dims = tuple(range(1, self.mean.dim()))
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    def sample(self):
        x = self.mean + self.std * torch.randn_like(self.mean)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.zeros(self.mean.shape[0], device=self.mean.device)
        else:
            if other is None:
                return 0.5 * torch.sum(
                    torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
                    dim=self.sum_dims,
                )
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar,
                    dim=self.sum_dims,
                )

    def nll(self, sample, dims=None):
        if self.deterministic:
            return torch.Tensor([0.0])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims or self.sum_dims,
        )

    def mode(self):
        return self.mean


def kl_to_standard_normal(
    mean: torch.Tensor, logvar: torch.Tensor, reduction: str = "mean"
) -> torch.Tensor:
    """
    KL( N(mean, exp(logvar)) || N(0, I) ), reduced over every dimension but the batch.

    ``reduction="mean"`` averages over latent tokens and channels instead of summing over
    them. With ~50k tokens x 2048 channels the summed KL is O(1e8) and cannot be balanced
    against a reconstruction loss of order one by any sane weight, so the per-element mean
    is the useful parametrisation here.
    """

    kl = 0.5 * (mean.pow(2) + logvar.exp() - 1.0 - logvar)
    dims = tuple(range(1, kl.dim()))
    if reduction == "mean":
        return kl.mean(dim=dims)
    elif reduction == "sum":
        return kl.sum(dim=dims)
    raise ValueError(f"Unknown reduction '{reduction}'. Expected 'mean' or 'sum'.")


class GaussianLatentHead(nn.Module):
    """
    VAE bottleneck applied to the global latent tokens.

    Maps the encoder output to a diagonal Gaussian posterior q(z|x) and returns a
    reparameterised sample (the posterior mean outside of training) together with the
    distribution parameters, so that a KL term can be computed by the loss modules.

    With ``identity_init`` the head starts out as the identity on the mean branch and at a
    constant, small posterior variance. That makes the model at step 0 numerically identical
    to the deterministic auto-encoder it is fine-tuned from (up to the injected noise), which
    a default-initialised Linear would instead scramble.
    """

    def __init__(
        self,
        dim: int,
        logvar_init: float = -5.0,
        identity_init: bool = True,
        logvar_min: float = -30.0,
        logvar_max: float = 20.0,
    ):
        super().__init__()

        self.dim = dim
        self.logvar_min = logvar_min
        self.logvar_max = logvar_max
        self.to_mean_logvar = nn.Linear(dim, 2 * dim, bias=True)

        with torch.no_grad():
            nn.init.zeros_(self.to_mean_logvar.bias)
            self.to_mean_logvar.bias[dim:].fill_(logvar_init)
            if identity_init:
                nn.init.zeros_(self.to_mean_logvar.weight)
                self.to_mean_logvar.weight[:dim].copy_(torch.eye(dim))

    def forward(
        self, x: torch.Tensor, sample: bool | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: latent tokens, [batch, num_tokens, dim]
            sample: draw from the posterior instead of taking its mode. Defaults to the
                module's training flag.
        Returns:
            (z, mean, logvar); z carries the dtype of ``x``, mean/logvar are fp32.
        """

        params = self.to_mean_logvar(x).float()
        mean, logvar = torch.chunk(params, 2, dim=-1)
        logvar = torch.clamp(logvar, self.logvar_min, self.logvar_max)

        sample = self.training if sample is None else sample
        z = mean + torch.exp(0.5 * logvar) * torch.randn_like(mean) if sample else mean

        return z.to(x.dtype), mean, logvar


class LatentInterpolator(nn.Module):
    """
    Code taken and adapted from: https://github.com/Jiawei-Yang/DeTok/tree/main
    """

    def __init__(
        self,
        gamma,
        dim,
        use_additive_noise=False,
        deterministic=False,
        saturate_encodings=None,
    ):
        super().__init__()

        assert deterministic or saturate_encodings is None, (
            "Cannot use saturate_encodings without deterministic"
        )
        self.gamma = gamma
        self.saturate_encodings = saturate_encodings
        self.use_additive_noise = use_additive_noise
        self.deterministic = deterministic
        self.mean_and_var = nn.Sequential(
            nn.Linear(dim, 2 * dim, bias=False),
            SaturateEncodings(saturate_encodings)
            if saturate_encodings is not None
            else nn.Identity(),
        )

    def interpolate_with_noise(self, z, batch_size=1, sampling=False, noise_level=-1):
        assert batch_size == 1, (
            "Given how we chunk in assimilate_local, dealing with batch_size greater than 1 is not "
            + "supported at the moment"
        )
        # a fresh distribution per call: this is invoked once per healpix chunk and the
        # caller collects the results in a list, so a single instance stored on the module
        # would leave every entry aliasing the last chunk's parameters
        diag_gaussian = DiagonalGaussianDistribution(
            deterministic=self.deterministic, channel_dim=-1
        )
        diag_gaussian.reset_parameters(self.mean_and_var(z))
        z_latents = diag_gaussian.sample() if sampling else diag_gaussian.mean

        if self.training and self.gamma > 0.0:
            device = z_latents.device
            s = z_latents.shape
            if noise_level > 0.0:
                noise_level_tensor = torch.full((batch_size,), noise_level, device=device)
            else:
                noise_level_tensor = torch.rand(batch_size, device=device)
            noise = torch.randn(s, device=device) * self.gamma
            if self.use_additive_noise:
                z_latents = z_latents + noise_level_tensor * noise
            else:
                z_latents = (1 - noise_level_tensor) * z_latents + noise_level_tensor * noise

        # the distribution keeps fp32 parameters for a stable KL; hand the latents back in
        # the dtype the surrounding (autocast) engines expect
        return z_latents.to(z.dtype), diag_gaussian
