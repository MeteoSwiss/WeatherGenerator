# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import torch

from weathergen.model.utils import check_encoder_checkpoint


class EMAModel:
    """
    Taken and modified from https://github.com/NVlabs/edm2/tree/main
    """

    @torch.no_grad()
    def __init__(
        self,
        model,
        empty_model,
        halflife_steps=float("inf"),
        rampup_ratio=0.09,
        is_model_sharded=False,
    ):
        self.original_model = model
        self.halflife_steps = halflife_steps
        self.rampup_ratio = rampup_ratio
        self.ema_model = empty_model
        self.is_model_sharded = is_model_sharded
        self.batch_size = 1
        # Build a name → param map once
        self.src_params = {
            name.removeprefix("module."): param
            for name, param in self.original_model.named_parameters()
        }
        teacher_params = dict(self.ema_model.named_parameters())
        source_encoders = {name for name in self.src_params if name.startswith("encoders.")}
        teacher_encoders = {name for name in teacher_params if name.startswith("encoders.")}
        if source_encoders != teacher_encoders:
            raise ValueError("EMA teacher must contain every named student encoder parameter.")
        for name, param in teacher_params.items():
            source = self.src_params.get(name)
            if source is None or source.shape != param.shape:
                raise ValueError(f"EMA parameter {name!r} must match the student's name and shape.")

        self.reset()

    @torch.no_grad()
    def reset(self):
        """
        This function resets the EMAModel to be the same as the Model.

        It operates via the state_dict to be able to deal with sharded tensors in case
        FSDP2 is used.
        """
        device = next(iter(self.src_params.values())).device
        if any(param.is_meta for param in self.ema_model.parameters()):
            self.ema_model.to_empty(device=device)
        else:
            self.ema_model.to(device)
        for p in self.ema_model.parameters():
            p.requires_grad = False
        maybe_sharded_sd = {
            key.removeprefix("module."): value
            for key, value in self.original_model.state_dict().items()
        }
        check_encoder_checkpoint(self.ema_model, maybe_sharded_sd)
        self.ema_model.load_state_dict(maybe_sharded_sd, strict=False, assign=False)
        self.ema_model.eval()

    def requires_grad_(self, flag: bool):
        for p in self.ema_model.parameters():
            p.requires_grad = flag

    def get_current_beta(self, cur_step: int) -> float:
        """
        Get current EMA beta value for monitoring.

        The beta value determines how much the teacher model is updated towards
        the student model at each step. Higher beta means slower teacher updates.

        Args:
            cur_step: Current training step (typically istep * batch_size).

        Returns:
            Current EMA beta value.
        """
        halflife_steps = self.halflife_steps
        if self.rampup_ratio is not None:
            halflife_steps = min(halflife_steps, cur_step * self.rampup_ratio)
        beta = 0.5 ** (self.batch_size / max(halflife_steps, 1e-6))
        return beta

    @torch.no_grad()
    def update(self, cur_step, batch_size):
        # ensure model remains sharded
        if self.is_model_sharded:
            self.ema_model.reshard()
        # determine correct interpolation params
        self.batch_size = batch_size
        beta = self.get_current_beta(cur_step)

        for name, p_ema in self.ema_model.named_parameters():
            p_ema.lerp_(self.src_params[name], 1.0 - beta)

    @torch.no_grad()
    def forward_eval(self, *args, **kwargs):
        self.ema_model.eval()
        out = self.ema_model(*args, **kwargs)
        return out

    def state_dict(self):
        return self.ema_model.state_dict()

    def load_state_dict(self, state, **kwargs):
        state = {key.removeprefix("module."): value for key, value in state.items()}
        check_encoder_checkpoint(self.ema_model, state)
        self.ema_model.load_state_dict(state, **kwargs)
