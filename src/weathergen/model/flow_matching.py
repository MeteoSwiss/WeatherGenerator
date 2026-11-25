# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Simple flow matching (rectified flow) on the decoder outputs.

The decoder becomes a conditional velocity field v(x_t, t | latent, coords) for the target values
(in normalised space). Per target point:

    x0 ~ N(0, 1),  x1 = target,  t ~ U(0, 1),  x_t = (1 - t) x0 + t x1,  v* = x1 - x0

The noisy values x_t and the flow time t enter the decoder through FlowMatchingEmbed, which is
added to the target coordinate embedding (query tokens). Its output layer starts at zero, so a
model warm-started from a deterministic checkpoint initially ignores x_t and t.

Training: the model returns x0 + v, so the existing MSE loss MSE(x0 + v, x1) equals the flow
matching loss MSE(v, x1 - x0). The logged training loss is therefore a velocity loss and is not
on the same scale as the validation loss.

Sampling: Euler integration from x0 ~ N(0, 1) over num_sampling_steps, rerunning only the decoder
(encoder and forecasting engine run once). num_members independent samples form an ensemble.
"""

import math

import torch
import torch.nn as nn


class ZeroInitLinear(nn.Linear):
    """Linear layer that is (re-)initialised to zero by reset_parameters()."""

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class FlowMatchingEmbed(nn.Module):
    """Embeds noisy target values x_t [P, C] and flow time t [P, 1] into the query token space."""

    def __init__(self, num_channels: int, dim_out: int, num_freqs: int = 16, dim_hidden: int = 256):
        super().__init__()
        self.num_freqs = num_freqs
        self.lin_in = nn.Linear(num_channels + 2 * num_freqs, dim_hidden)
        self.act = nn.SiLU()
        self.lin_out = ZeroInitLinear(dim_hidden, dim_out)

    def time_features(self, t: torch.Tensor) -> torch.Tensor:
        freqs = torch.pow(2.0, torch.arange(self.num_freqs, device=t.device, dtype=t.dtype))
        arg = t * freqs * math.pi
        return torch.cat([torch.sin(arg), torch.cos(arg)], dim=-1)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = torch.cat([x_t, self.time_features(t)], dim=-1)
        return self.lin_out(self.act(self.lin_in(h)))

    def reset_parameters(self) -> None:
        self.lin_in.reset_parameters()
        self.lin_out.reset_parameters()


def flow_matching_targets(batch) -> dict:
    """Target values per stream and output step, aligned with the source samples.

    Uses the same source -> target correspondence as LossPhysical: the target sample whose
    meta_info[stream].global_params["idx"] equals the source sample's "correspondence".

    Returns: {stream_name: {step: [target_tokens tensor per source sample]}}
    """
    source = batch.get_source_samples()
    target = batch.get_target_samples()
    stream_names = list(source.samples[0].streams_data.keys())
    out = {}
    for stream_name in stream_names:
        out[stream_name] = {}
        tgt_idx = [t.meta_info[stream_name].global_params["idx"] for t in target.samples]
        for step in batch.get_output_idxs():
            per_source = []
            for s in source.samples:
                corr = list(s.meta_info.values())[0].global_params.get("correspondence", -1)
                matches = [j for j, idx in enumerate(tgt_idx) if idx == corr]
                assert len(matches) == 1, "flow matching needs a unique target per source sample"
                per_source.append(
                    target.samples[matches[0]].streams_data[stream_name].target_tokens[step]
                )
            out[stream_name][step] = per_source
    return out
