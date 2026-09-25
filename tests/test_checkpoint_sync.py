from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

pytest.importorskip("flash_attn", reason="required to import the native trainer")

from weathergen.train.trainer import Trainer  # noqa: E402


def _save_checkpoint(rank, path, writer_ready, peer_ready, allow_write, peer_returned):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{path}/rendezvous",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    trainer = Trainer.__new__(Trainer)
    trainer.cf = SimpleNamespace(
        with_ddp=True, with_fsdp=False, general=SimpleNamespace(run_id="probe")
    )
    trainer.training_cfg = SimpleNamespace(num_mini_epochs=1)
    trainer.model = torch.nn.Linear(1, 1, bias=False)
    trainer.model.weight.data.fill_(rank + 1)
    trainer.ema_model = None
    try:
        with (
            patch("weathergen.train.trainer.config.get_path_model", return_value=path),
            patch("weathergen.train.trainer.config.save"),
        ):
            if rank == 0:
                writer_ready.set()
                assert allow_write.wait(20), "writer was never released"
            else:
                peer_ready.set()
            trainer.save_model(1)
        if rank == 1:
            peer_returned.set()
            state = torch.load(path / "probe_chkpt00001.chkpt", weights_only=True)
            torch.testing.assert_close(state["weight"], torch.ones(1, 1))
    finally:
        dist.destroy_process_group()


def test_checkpoint_is_readable_before_nonroot_rank_can_exit(tmp_path):
    context = mp.get_context("spawn")
    writer_ready, peer_ready, allow_write, peer_returned = [context.Event() for _ in range(4)]
    workers = mp.spawn(
        _save_checkpoint,
        args=(tmp_path, writer_ready, peer_ready, allow_write, peer_returned),
        nprocs=2,
        join=False,
    )
    try:
        assert writer_ready.wait(60), "writer did not start"
        assert peer_ready.wait(60), "peer did not start"
        assert not peer_returned.wait(1), "peer exited before rank zero wrote the checkpoint"
    finally:
        allow_write.set()
        while not workers.join(timeout=30):
            pass
    assert peer_returned.is_set()
