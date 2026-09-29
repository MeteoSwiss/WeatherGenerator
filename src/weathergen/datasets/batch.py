"""
Data structures for student-teacher multi-view training.

Provides clean separation between:
  - Model data (StreamData objects containing tensors)
  - View metadata (spatial masks, strategies, relationships)
"""

import copy
from dataclasses import dataclass

import numpy as np
import torch

from weathergen.common.config import Config
from weathergen.datasets.stream_data import StreamData


@dataclass
class SampleMetaData:
    # sample parameters (masking)
    params: Config | dict

    mask: torch.Tensor | None = None

    global_params: dict | None = None


class Sample:
    # keys: stream name, values: SampleMetaData
    meta_info: dict[str, SampleMetaData]

    # Common forecast-grid visibility and view correspondence.
    view_meta: SampleMetaData | None

    # data for all streams
    # keys: stream_name, values: StreamData
    streams_data: dict[str, StreamData | None]

    def pin_memory(self):
        """Pin all tensors in this Sample to CPU pinned memory"""

        for stream in self.streams_data.values():
            if stream is not None:
                stream.pin_memory()

        for metadata in (*self.meta_info.values(), self.view_meta):
            if metadata is not None and metadata.mask is not None:
                metadata.mask = metadata.mask.pin_memory()

        return self

    def __init__(self, stream_names: list[str]) -> None:
        self.meta_info = {}
        self.view_meta = None

        self.streams_data = dict.fromkeys(stream_names)

    def to_device(self, device) -> None:
        for metadata in (*self.meta_info.values(), self.view_meta):
            if metadata is not None and metadata.mask is not None:
                metadata.mask = metadata.mask.to(device, non_blocking=True)

        for stream in self.streams_data.values():
            if stream is not None:
                stream.to_device(device)

    def is_empty(self) -> bool:
        """
        Check if sample is empty
        """
        return all(s is None or s.empty() for s in self.streams_data.values())

    def is_nan(self) -> bool:
        """
        Check if sample is all NaN
        """
        return all(s is not None and s.nan() for s in self.streams_data.values())

    def sources_empty(self) -> bool:
        """
        Check if sources for sample are empty
        """
        return all(s is None or s.source_empty() for s in self.streams_data.values())

    def sources_nan(self) -> bool:
        """
        Check if sources for sample are all NaN
        """
        streams = [s for s in self.streams_data.values() if s is not None and not s.source_empty()]
        return bool(streams) and all(s.source_nan() for s in streams)

    def targets_empty(self) -> bool:
        """
        Check if targets for sample are empty
        """
        return all(s is None or s.target_empty() for s in self.streams_data.values())

    def targets_nan(self) -> bool:
        """
        Check if targets for sample are all NaN
        """
        return all(s is not None and s.target_nan() for s in self.streams_data.values())

    def add_stream_data(self, stream_name: str, stream_data: StreamData) -> None:
        """
        Add data for stream @stream_name to sample
        """
        self.streams_data[stream_name] = stream_data

    def add_meta_info(self, stream_name: str, meta_info: SampleMetaData) -> None:
        """
        Add metadata for stream @stream_name to sample
        """
        self.meta_info[stream_name] = meta_info

    def get_stream_data(self, stream_name: str) -> StreamData:
        """
        Get data for stream @stream_name from sample
        """
        return self.streams_data[stream_name]

    def get_num_source_steps(self) -> int:
        """Get the minimum input length among available streams."""
        return min(
            (s.get_num_source_steps() for s in self.streams_data.values() if s is not None),
            default=0,
        )

    def get_num_target_steps(self) -> int:
        """Get the minimum target length among available streams."""
        return min(
            (s.get_num_target_steps() for s in self.streams_data.values() if s is not None),
            default=0,
        )


class BatchSamples:
    """
    Container for source or target samples
    """

    samples: list[Sample]
    tokens_lens: torch.Tensor | None
    output_steps: int
    output_idxs: list[int]
    device: str | None

    def __init__(
        self, stream_names: list[str], num_samples: int, output_steps, output_idxs
    ) -> None:
        self.samples = [Sample(stream_names) for _ in range(num_samples)]
        self.tokens_lens = None
        self.output_steps = output_steps
        self.output_idxs = output_idxs
        self.device = None
        self.encoder_batches: dict[str, BatchSamples] = {}
        self.encoder_name: str | None = None
        self.healpix_level: int | None = None
        self.coverage: torch.Tensor | None = None

    def __len__(self) -> int:
        return len(self.samples)

    def to_device(self, device):
        for sample in self.samples:
            sample.to_device(device)

        if self.tokens_lens is not None:
            self.tokens_lens = self.tokens_lens.to(device, non_blocking=True)
        if self.coverage is not None:
            self.coverage = self.coverage.to(device, non_blocking=True)
        for child in self.encoder_batches.values():
            child.to_device(device)

        self.device = device

        return self

    def get_samples(self) -> list[Sample]:
        return self.samples

    def get_subset(self, subset: list | None = None):
        if subset is None:
            return self
        bs = copy.copy(self)
        bs.samples = copy.deepcopy([self.samples[i] for i in subset])
        if self.tokens_lens is not None:
            bs.tokens_lens = self.tokens_lens[:, subset]
        if self.coverage is not None:
            bs.coverage = self.coverage[:, subset]
        bs.encoder_batches = {
            name: child.get_subset(subset) for name, child in self.encoder_batches.items()
        }
        return bs

    def get_num_source_steps(self) -> int:
        """
        Get number of input/source steps from smallest of all available streams
        """
        if self.encoder_batches:
            return min(child.get_num_source_steps() for child in self.encoder_batches.values())
        if self.coverage is not None:
            return self.coverage.shape[0]
        return self.samples[0].get_num_source_steps() if self.samples else 0

    def get_num_target_steps(self) -> int:
        """
        Get number of target steps from smallest of all available streams
        """
        return self.samples[0].get_num_target_steps()

    def get_output_idxs(self) -> int:
        """
        Get forecast indices
        """
        return self.output_idxs

    def get_output_len(self) -> int:
        """
        Get length of output
        """
        return self.output_steps

    def get_device(self) -> str | torch.device:
        """
        Get device of tensors in the batch
        """
        return self.device

    def sources_empty(self) -> bool:
        """
        Check if sources for all samples are empty
        """
        if self.encoder_batches:
            return all(child.sources_empty() for child in self.encoder_batches.values())
        if self.coverage is not None:
            # Coverage precedes training masking: learned queries remain valid when fully masked.
            return not bool(self.coverage.any())
        return all(s.sources_empty() for s in self.samples)

    def targets_empty(self) -> bool:
        """
        Check if targets for all samples are empty
        """
        return all(s.targets_empty() for s in self.samples)

    def sources_nan(self) -> bool:
        """
        Check if sources for all samples are all NaN
        """
        if self.encoder_batches:
            children = [
                child for child in self.encoder_batches.values() if not child.sources_empty()
            ]
            return bool(children) and all(child.sources_nan() for child in children)
        samples = [s for s in self.samples if not s.sources_empty()]
        return bool(samples) and all(s.sources_nan() for s in samples)

    def targets_nan(self) -> bool:
        """
        Check if targets for all samples are all NaN
        """
        return all(s.targets_nan() for s in self.samples)

    def pin_memory(self):
        """Pin all tensors in this batch to CPU pinned memory"""

        for sample in self.samples:
            sample.pin_memory()

        if self.tokens_lens is not None:
            self.tokens_lens = self.tokens_lens.pin_memory()
        if self.coverage is not None:
            self.coverage = self.coverage.pin_memory()
        for child in self.encoder_batches.values():
            child.pin_memory()

        return self


class ModelBatch:
    """
    Container for all data and metadata for one training batch.
    """

    # source samples (for model)
    source_samples: BatchSamples

    # target samples (for TargetAuxCalculator)
    target_samples: BatchSamples

    # index of corresponding target (for source samples) or source (for target samples)
    # these are in 1-to-1 corresponding for classical training modes (e.g. MTM, forecasting) but
    # can be more complex for strategies like student-teacher training
    source2target_matching_idxs: np.typing.NDArray[np.int32]
    target2source_matching_idxs: np.typing.NDArray[np.int32]

    # indices of valid outputs
    output_idxs: list[int]

    # device of the tensors in the batch
    device: str | torch.device

    def __init__(
        self,
        stream_names: list[str],
        num_source_samples: int,
        num_target_samples: int,
        output_offset,
        output_steps,
    ) -> None:
        """ """

        # define forecast indices
        self.output_offset = output_offset
        self.output_steps = output_steps
        self.output_idxs = list(range(output_offset, output_steps))

        self.source_samples = BatchSamples(
            stream_names, num_source_samples, output_steps, self.output_idxs
        )
        self.target_samples = BatchSamples(
            stream_names, num_target_samples, output_steps, self.output_idxs
        )

        self.source2target_matching_idxs = np.full(num_source_samples, -1, dtype=np.int32)
        self.target2source_matching_idxs = [[] for _ in range(num_target_samples)]

    def pin_memory(self):
        """Pin all tensors in this batch to CPU pinned memory"""

        # pin source samples
        self.source_samples.pin_memory()

        # pin target samples
        self.target_samples.pin_memory()

        return self

    def to_device(self, device):  # -> ModelBatch
        """
        Move batch to device
        """

        self.source_samples.to_device(device)
        self.target_samples.to_device(device)

        self.device = device

        return self

    def add_source_stream(
        self,
        source_sample_idx: int,
        target_sample_idx: int,
        stream_name: str,
        stream_data: StreamData,
        source_meta_info: SampleMetaData,
    ) -> None:
        """
        Add data for one stream to sample @source_sample_idx
        """
        self.source_samples.samples[source_sample_idx].add_stream_data(stream_name, stream_data)

        # add the meta_info
        self.source_samples.samples[source_sample_idx].add_meta_info(stream_name, source_meta_info)

        self.source2target_matching_idxs[source_sample_idx] = target_sample_idx

    def add_target_stream(
        self,
        target_sample_idx: int,
        source_sample_idx: int | list[int],
        stream_name: str,
        stream_data: StreamData,
        target_meta_info: SampleMetaData,
    ) -> None:
        """
        Add data for one stream to sample @target_sample_idx
        """
        self.target_samples.samples[target_sample_idx].add_stream_data(stream_name, stream_data)

        # add the meta_info -- for target we have different
        self.target_samples.samples[target_sample_idx].add_meta_info(stream_name, target_meta_info)

        self.target2source_matching_idxs[target_sample_idx] = source_sample_idx

    def is_empty(self):
        """
        Check if batch is empty
        """
        return self.source_samples.sources_empty() or self.target_samples.targets_empty()

    def is_nan(self):
        """
        Check if batch is all NaN
        """
        return self.source_samples.sources_nan() or self.target_samples.targets_nan()

    def sources_empty(self):
        """
        Check if batch sources are empty
        """
        return self.source_samples.sources_empty()

    def sources_nan(self):
        """
        Check if batch sources are all NaN
        """
        return self.source_samples.sources_nan()

    def targets_empty(self):
        """
        Check if batch targets are empty
        """
        return self.target_samples.targets_empty()

    def targets_nan(self):
        """
        Check if batch targets are all NaN
        """
        return self.target_samples.targets_nan()

    def len_sources(self) -> int:
        """
        Number of source samples
        """
        return len(self.source_samples)

    def len_targets(self) -> int:
        """
        Number of target samples
        """
        return len(self.target_samples)

    def get_source_sample(self, idx: int) -> Sample:
        """
        Get a source sample
        """
        return self.source_samples.samples[idx]

    def get_source_samples(self, subset: list | None = None) -> BatchSamples:
        """
        Get source samples
        """
        return self.source_samples.get_subset(subset)

    def get_target_sample(self, idx: int) -> Sample:
        """
        Get a target sample
        """
        return self.target_samples.samples[idx]

    def get_target_samples(self, subset: list | None = None) -> BatchSamples:
        """
        Get target samples
        """
        return self.target_samples.get_subset(subset)

    def get_source_idx_for_target(self, target_idx: int) -> int:
        """
        Get index of source sample for a given target sample index
        """
        return int(self.target2source_matching_idxs[target_idx])

    def get_target_idx_for_source(self, source_idx: int) -> int:
        """
        Get index of target sample for a given source sample index
        """
        return int(self.source2target_matching_idxs[source_idx])

    def get_output_idxs(self) -> int:
        """
        Get valid output steps
        """
        return self.output_idxs

    def get_output_len(self) -> int:
        """
        Get length of output
        """
        return self.output_steps

    def get_device(self) -> str | torch.device:
        """
        Get device of tensors in the batch
        """
        return self.device

    def get_num_source_steps(self) -> int:
        """
        Get number of input/source steps from smallest of all available streams
        """
        return self.source_samples.get_num_source_steps()

    def get_num_target_steps(self) -> int:
        """
        Get number of target steps from smallest of all available streams
        """
        return self.target_samples.get_num_target_steps()
