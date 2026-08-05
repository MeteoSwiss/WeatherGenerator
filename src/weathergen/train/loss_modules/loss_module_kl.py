# ruff: noqa: T201

# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging

import torch
from omegaconf import DictConfig

from weathergen.model.parametrised_prob_dist import kl_to_standard_normal
from weathergen.train.loss_modules.loss_module_base import LossModuleBase, LossValues
from weathergen.utils.train_logger import Stage

_logger = logging.getLogger(__name__)


class LossKL(LossModuleBase):
    """
    VAE regularisation of the global latent space: KL( q(z|x) || N(0, I) ).

    Reads the posterior parameters that ``EncoderModule`` attaches to the model output when
    ``latent_vae.enabled`` is set, and returns them as a loss term so that the existing
    weighting, history and metric-logging machinery applies unchanged.

    Configured as a regular loss term, e.g.::

        "kl": { type: LossKL, weight: 1.0e-5, target_and_aux_calc: Physical, loss_fcts: {} }

    The term is inert (zero loss, NaN metric) when the VAE bottleneck is disabled, so the
    same config can be reused for non-VAE runs.
    """

    def __init__(
        self,
        cf: DictConfig,
        mode_cfg: DictConfig,
        stage: Stage,
        device: str,
        **loss_fcts: dict,
    ):
        LossModuleBase.__init__(self)
        self.cf = cf
        self.stage = stage
        self.device = device
        self.name = "LossKL"

        vae_cfg = cf.get("latent_vae", None) or {}
        # per-element KL floor: below it the term is switched off dimension-wise, which keeps
        # a few latent channels from being driven to the prior and collapsing (Kingma et al.,
        # "Improving Variational Inference with Inverse Autoregressive Flow", 2016)
        self.free_bits = vae_cfg.get("free_bits", 0.0)
        # linear beta ramp; the encoder starts deterministic, so applying the full weight
        # from step 0 tends to wreck reconstruction before the decoder can adapt
        self.warmup_steps = vae_cfg.get("kl_warmup_steps", 0)
        self.reduction = vae_cfg.get("kl_reduction", "mean")

        self.loss_fcts = []

    def _beta_scale(self) -> float:
        """Linear warm-up factor in [0, 1] applied on top of the configured term weight."""

        if self.warmup_steps <= 0 or self.stage != "train":
            return 1.0
        istep = self.cf.get("general", {}).get("istep", 0)
        return min(1.0, (istep + 1) / self.warmup_steps)

    def compute_loss(self, preds, targets, **kwargs) -> LossValues:
        latent = preds.latent[0] if preds.latent else {}
        mean = latent.get("posterior_mean", None)
        logvar = latent.get("posterior_logvar", None)

        if mean is None or logvar is None:
            # VAE bottleneck disabled: stay out of the way rather than crash
            nan = torch.tensor(torch.nan, device=self.device)
            return LossValues(
                loss=torch.zeros(1, device=self.device),
                losses_all={"kl": nan, "beta_scale": nan},
                stddev_all=None,
            )

        if self.free_bits > 0.0:
            kl_elem = 0.5 * (mean.pow(2) + logvar.exp() - 1.0 - logvar)
            dims = tuple(range(1, kl_elem.dim()))
            kl = kl_elem.clamp(min=self.free_bits).mean(dim=dims)
        else:
            kl = kl_to_standard_normal(mean, logvar, reduction=self.reduction)

        # report the raw KL, optimise the warmed-up one, so the metric stays comparable
        # across the ramp
        kl = kl.mean()
        beta_scale = self._beta_scale()

        return LossValues(
            loss=beta_scale * kl,
            losses_all={
                # LossCalculator nests these under self.name, giving LossKL.kl / LossKL.beta_scale
                "kl": kl.detach(),
                "beta_scale": torch.tensor(beta_scale, device=self.device),
            },
            stddev_all=None,
        )
