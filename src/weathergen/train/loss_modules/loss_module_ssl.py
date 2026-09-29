# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import torch
import torch.nn.functional as F
from omegaconf import DictConfig

import weathergen.train.loss_modules.loss_functions as loss_fns
from weathergen.train.loss_modules.loss_module_base import LossModuleBase, LossValues
from weathergen.utils.train_logger import Stage


class LossLatentSSLStudentTeacher(LossModuleBase):
    """
    Manages and computes the overall loss for a WeatherGenerator model pretraining using
    DINO/iBOT/JEPA/BYOL style losses.

    This class handles the initialization and application of various loss functions,
    It provides both the main loss for backpropagation and detailed loss metrics for logging.
    """

    valid_loss_names = {"DINO", "iBOT", "JEPA"}

    def __init__(self, cf: DictConfig, mode_cfg: DictConfig, stage: Stage, device: str, **losses):
        LossModuleBase.__init__(self)
        self.cf = cf
        self.stage = stage
        self.device = device
        self.num_class_tokens = cf.num_class_tokens
        self.name = "LossLatentSSLStudentTeacher"

        # Dynamically load loss functions based on configuration and stage
        self.losses = {
            name: (local_conf["weight"], get_loss_function_ssl(name), local_conf["loss_extra_args"])
            for name, local_conf in losses.items()
        }

    def compute_loss(self, preds, targets, metadata) -> LossValues:
        # gradient loss
        loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        # initialize dictionaries for detailed loss tracking and standard deviation statistics
        # create tensor for each stream
        losses_all: dict[str, float] = {loss: 0.0 for loss in self.losses}

        _, output_info, _, _ = metadata

        preds = preds.latent[0]  # [0] because we always want the first fstep
        target_info = targets.aux_outputs
        targets = targets.latent
        target_ids = [info.global_params["idx"] for info in target_info]
        target2source_matching_idxs = [
            [
                i
                for i, info in enumerate(output_info)
                if info.global_params["correspondence"] == target_id
            ]
            for target_id in target_ids
        ]

        for name, (weight, loss_fn, extra_args) in self.losses.items():
            for values, info in ((preds[name], output_info), (targets[name], target_info)):
                if values.ndim != 3 or values.shape[0] != len(info):
                    raise ValueError(f"{name}: latent batch axis must match view metadata.")
            if preds[name].shape[-1] != targets[name].shape[-1]:
                raise ValueError(f"{name}: student and teacher feature axes must match.")
            teacher_values, teacher_info = targets[name], target_info
            if name in ("JEPA", "iBOT"):
                teacher_by_id = {
                    info.global_params["idx"]: (value, info)
                    for value, info in zip(teacher_values, teacher_info, strict=True)
                    if name in info.global_params["loss"]
                }
                matched = [
                    teacher_by_id[info.global_params["correspondence"]]
                    for info in output_info
                    if name in info.global_params["loss"]
                ]
                teacher_values = torch.stack([value for value, _ in matched])
                teacher_info = [info for _, info in matched]
            preds_for_loss = self.gather_preds_for_loss(
                name, preds[name], output_info, target2source_matching_idxs
            )
            targets_for_loss = self.gather_targets_for_loss(
                name, teacher_values, teacher_info, target2source_matching_idxs
            )

            loss_value = loss_fn(**preds_for_loss, **targets_for_loss, **extra_args).mean()
            loss = loss + (weight * loss_value)
            losses_all[name] = loss_value.item()

        return LossValues(loss=loss, losses_all=losses_all, stddev_all={})

    def _patch_masks(self, metadata, name):
        masks = torch.stack([info.mask for info in metadata if name in info.global_params["loss"]])
        if masks.shape[1:] != (12 * 4**self.cf.fe_healpix_level,):
            raise ValueError(f"{name}: visibility masks must use the fused forecast grid.")
        if self.cf.fe_num_queries > 1:
            masks = masks.repeat_interleave(self.cf.fe_num_queries, dim=-1)
        return masks.unsqueeze(1)

    def gather_preds_for_loss(self, name, preds, metadata, target2source_matching_idxs):
        if name in ("JEPA", "iBOT"):
            values = torch.stack(
                [
                    p
                    for p, info in zip(preds, metadata, strict=True)
                    if name in info.global_params["loss"]
                ]
            )
            prefix = self.num_class_tokens if name == "iBOT" else 0
            result = {
                "student_patches_masked": values[:, prefix:],
                "student_masks": self._patch_masks(metadata, name),
            }
            if name == "iBOT":
                result["student_class_masked"] = values[:, :prefix]
            return result
        elif name == "DINO":
            local2global_dino_student = []
            for student_indices in target2source_matching_idxs:
                local_preds = [
                    preds[sidx]
                    for sidx in student_indices
                    if "DINO" in metadata[sidx].global_params["loss"]
                    and metadata[sidx].global_params["relationship"] != "identity"
                ]
                local2global_dino_student.append(local_preds)
            local2global_dino_student = [
                torch.stack(latents, dim=0)
                for latents in zip(*local2global_dino_student, strict=False)
            ]
            return {
                "local2global_dino_student": local2global_dino_student,
                "global2global_dino_student": torch.stack(
                    [
                        p
                        for p, info in zip(preds, metadata, strict=True)
                        if "DINO" in info.global_params["loss"]
                        and info.global_params["relationship"] == "identity"
                    ],
                    dim=0,
                ),
            }
        else:
            raise NotImplementedError(
                f"{name} is not an implemented loss for the LossLatentSSLStudentTeacher"
            )

    def gather_targets_for_loss(self, name, targets, metadata, target2source_matching_idxs):
        if name in ("JEPA", "iBOT"):
            values = torch.stack(
                [
                    p
                    for p, info in zip(targets, metadata, strict=True)
                    if name in info.global_params["loss"]
                ]
            )
            prefix = self.num_class_tokens if name == "iBOT" else 0
            result = {
                "teacher_patches_masked": values[:, prefix:],
                "teacher_masks": self._patch_masks(metadata, name),
            }
            if name == "iBOT":
                result["teacher_class_masked"] = values[:, :prefix]
            return result
        elif name == "DINO":
            return {
                "local2global_dino_teacher": targets,
                "global2global_dino_teacher": targets.flip(0),
            }
        else:
            raise NotImplementedError(
                f"{name} is not an implemented loss for the LossLatentSSLStudentTeacher"
            )


def jepa_loss(student_patches_masked, student_masks, teacher_patches_masked, teacher_masks):
    # TODO remove as we deal with batch dimension
    assert teacher_masks.shape[0] == 1 or teacher_masks.shape[0] == student_masks.shape[0]
    student_masks = student_masks.squeeze(dim=1)
    teacher_masks = teacher_masks.squeeze(dim=1)
    masks_weight = (
        (1 / student_masks.sum(-1).clamp(min=1.0))
        .unsqueeze(-1)
        .expand_as(student_masks)  # [student_masks_flat]
    )

    mask = torch.logical_and(teacher_masks, torch.logical_not(student_masks))

    assert mask.shape[0] == student_patches_masked.shape[0], (
        "mask.shape[0], batch dimension, has to match batch dimension for student_patches_masked."
    )
    # expand/repeat teacher_masks to match number of student samples
    teacher_patches = teacher_patches_masked.expand((mask.shape[0], -1, -1))
    # compute loss
    loss = F.l1_loss(student_patches_masked[mask], teacher_patches[mask])
    loss = loss * masks_weight[mask]

    return loss.sum()  # / student_masks.shape[0]


def ibot_loss(
    student_patches_masked,
    student_masks,
    teacher_patches_masked,
    teacher_masks,
    student_class_masked,
    teacher_class_masked,
    student_temp,
):
    student_masks = student_masks.squeeze(dim=1)
    teacher_masks = teacher_masks.squeeze(dim=1)
    loss = loss_fns.masked_student_teacher_patch_softmax(
        student_patches_masked, teacher_patches_masked, student_masks, teacher_masks, student_temp
    )
    loss = loss + loss_fns.student_teacher_softmax(
        student_class_masked, teacher_class_masked, student_temp
    )
    return loss / 2


def dino_loss(
    local2global_dino_student,
    local2global_dino_teacher,
    global2global_dino_student,
    global2global_dino_teacher,
    student_temp,
):
    loss = loss_fns.student_teacher_global_softmax(
        local2global_dino_student, local2global_dino_teacher, student_temp
    ) + loss_fns.student_teacher_softmax(
        global2global_dino_student, global2global_dino_teacher, student_temp
    )
    return loss / 2


def get_loss_function_ssl(name):
    if name == "iBOT":
        return ibot_loss
    elif name == "DINO":
        return dino_loss
    elif name == "JEPA":
        return jepa_loss
    else:
        raise NotImplementedError(
            f"{name} is not an implemented loss for the LossLatentSSLStudentTeacher"
        )
