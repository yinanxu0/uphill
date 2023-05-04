
from dataclasses import asdict, dataclass
from typing import Optional, Tuple, Dict
import torch
import numpy as np
import torchaudio
from decimal import ROUND_HALF_UP


from uphill.core.utils import Seconds
from .utils import (
    compute_num_samples, perturb_num_samples
)

class AudioAugment:
    """
    Base class for all audio transforms that are going to be lazily applied on
    ``Recording`` during loading the audio into memory.

    Any ``AudioAugment`` can be used like a Python function, that expects two arguments:
    a numpy array of samples, and a sampling rate. E.g.:

        >>> fn = AudioAugment.from_dict(...)
        >>> new_audio = fn(audio, sampling_rate)

    Since we often use cuts of the original recording, they will refer to the timestamps
    of the augmented audio (which might be speed perturbed and of different duration).
    Each transform provides a helper method to recover the original audio timestamps:

        >>> # When fn does speed perturbation:
        >>> fn.reverse_timestamps(offset=5.055555, duration=10.1111111, sampling_rate=16000)
        ... (5.0, 10.0)

    Furthermore, ``AudioAugment`` can be easily (de)serialized to/from dict
    that contains its name and parameters.
    This enables storing recording and cut manifests with the transform info
    inside, avoiding the need to store the augmented recording version on disk.

    All audio transforms derived from this class are "automagically" registered,
    so that ``AudioAugment.from_dict()`` can "find" the right type given its name
    to instantiate a specific transform object.
    All child classes are expected to be decorated with a ``@dataclass`` decorator.
    """

    NAME_TO_TRANSFORM = {}
    TRANSFORM_TO_NAME = {}

    def __init_subclass__(cls, **kwargs):
        if cls.__name__ not in AudioAugment.NAME_TO_TRANSFORM:
            AudioAugment.NAME_TO_TRANSFORM[cls.__name__] = cls
            AudioAugment.TRANSFORM_TO_NAME[cls] = cls.__name__
        super().__init_subclass__(**kwargs)

    def to_dict(self) -> dict:
        data = asdict(self)
        return {"name": type(self).__name__, "kwargs": data}

    @staticmethod
    def from_dict(data: dict) -> "AudioAugment":
        assert (
            data["name"] in AudioAugment.NAME_TO_TRANSFORM
        ), f"Unknown transform type: {data['name']}"
        return AudioAugment.NAME_TO_TRANSFORM[data["name"]](**data["kwargs"])

    def __call__(self, samples: np.ndarray, sampling_rate: int) -> np.ndarray:
        """
        Apply transform.

        To be implemented in derived classes.
        """
        raise NotImplementedError

    def reverse_timestamps(
        self, offset: Seconds, duration: Optional[Seconds], sampling_rate: int
    ) -> Tuple[Seconds, Optional[Seconds]]:
        """
        Convert ``offset`` and ``duration`` timestamps to be adequate for the audio before the transform.
        Useful for on-the-fly augmentation when a particular chunk of audio needs to be read from disk.

        To be implemented in derived classes.
        """
        raise NotImplementedError


@dataclass
class Speed(AudioAugment):
    """
    Speed perturbation effect, the same one as invoked with `sox speed` in the command line.

    It resamples the signal back to the input sampling rate, so the number of output samples will
    be smaller or greater, depending on the speed factor.
    """

    factor: float
    
    def __post_init__(self):
        self._precompiled_resamplers: Dict[Tuple[int, int], torch.nn.Module] = {}
    
    def __call__(self, samples: np.ndarray, sampling_rate: int) -> np.ndarray:
        ## TODO: round(sampling_rate * self.factor) -> sampling_rate ???
        resampler_key = (round(sampling_rate * self.factor), sampling_rate)
        if resampler_key not in self._precompiled_resamplers:
            self._precompiled_resamplers[resampler_key] = torchaudio.transforms.Resample(
                *resampler_key
            )
        resampler = self._precompiled_resamplers[resampler_key]
        augmented = resampler(torch.from_numpy(samples))
        return augmented.numpy()

    def reverse_timestamps(
        self, offset: Seconds, duration: Optional[Seconds], sampling_rate: int
    ) -> Tuple[Seconds, Optional[Seconds]]:
        """
        This method helps estimate the original offset and duration for a recording
        before speed perturbation was applied.
        We need this estimate to know how much audio to actually load from disk during the
        call to ``load_audio()``.
        """
        start_sample = compute_num_samples(offset, sampling_rate)
        num_samples = (
            compute_num_samples(duration, sampling_rate)
            if duration is not None
            else None
        )
        start_sample = perturb_num_samples(start_sample, 1 / self.factor)
        num_samples = (
            perturb_num_samples(num_samples, 1 / self.factor)
            if num_samples is not None
            else None
        )
        return (
            start_sample / sampling_rate,
            num_samples / sampling_rate if num_samples is not None else None,
        )


@dataclass
class Resample(AudioAugment):
    """
    Resampling effect, the same one as invoked with `sox rate` in the command line.
    """

    source_sampling_rate: int
    target_sampling_rate: int

    def __post_init__(self):
        self.source_sampling_rate = int(self.source_sampling_rate)
        self.target_sampling_rate = int(self.target_sampling_rate)
        self.resampler = torchaudio.transforms.Resample(
            self.source_sampling_rate, self.target_sampling_rate
        )

    def __call__(self, samples: np.ndarray, *args, **kwargs) -> np.ndarray:
        if self.source_sampling_rate == self.target_sampling_rate:
            return samples

        if isinstance(samples, np.ndarray):
            samples = torch.from_numpy(samples)
        augmented = self.resampler(samples)
        return augmented.numpy()

    def reverse_timestamps(
        self, offset: Seconds, duration: Optional[Seconds], sampling_rate: int
    ) -> Tuple[Seconds, Optional[Seconds]]:
        """
        This method helps estimate the original offset and duration for a recording
        before resampling was applied.
        We need this estimate to know how much audio to actually load from disk during the
        call to ``load_audio()``.

        In case of resampling, the timestamps might change slightly when using non-trivial
        pairs of sampling rates, e.g. 16kHz -> 22.05kHz, because the number of samples in
        the resampled audio might actually correspond to incrementally larger/smaller duration.
        E.g. 16kHz, 235636 samples correspond to 14.72725s duration; after resampling to 22.05kHz,
        it is 324736 samples which correspond to 14.727256235827664s duration.
        """
        if self.source_sampling_rate == self.target_sampling_rate:
            return offset, duration

        old_num_samples = compute_num_samples(
            offset, self.source_sampling_rate, rounding=ROUND_HALF_UP
        )
        old_offset = old_num_samples / self.source_sampling_rate
        if duration is not None:
            old_num_samples = compute_num_samples(
                duration, self.source_sampling_rate, rounding=ROUND_HALF_UP
            )
            old_duration = old_num_samples / self.source_sampling_rate
        else:
            old_duration = None
        return old_offset, old_duration


@dataclass
class Tempo(AudioAugment):
    """Tempo perturbation effect, the same one as invoked with `sox tempo` in the command line.

    Compared to speed perturbation, tempo preserves pitch.
    It resamples the signal back to the input sampling rate, so the number of output samples will
    be smaller or greater, depending on the tempo factor.
    """

    factor: float

    def __call__(self, samples: np.ndarray, sampling_rate: int) -> np.ndarray:
        if isinstance(samples, np.ndarray):
            samples = torch.from_numpy(samples)
        augmented, _ = torchaudio.sox_effects.apply_effects_tensor(
            samples, sampling_rate, [["tempo", str(self.factor)]]
        )
        return augmented.numpy()

    def reverse_timestamps(
        self,
        offset: Seconds,
        duration: Optional[Seconds],
        sampling_rate: int,
    ) -> Tuple[Seconds, Optional[Seconds]]:
        """
        This method helps estimate the original offset and duration for a recording
        before tempo perturbation was applied.
        We need this estimate to know how much audio to actually load from disk during the
        call to ``load_audio()``.
        """
        start_sample = compute_num_samples(offset, sampling_rate)
        num_samples = (
            compute_num_samples(duration, sampling_rate)
            if duration is not None
            else None
        )
        start_sample = perturb_num_samples(start_sample, 1 / self.factor)
        num_samples = (
            perturb_num_samples(num_samples, 1 / self.factor)
            if num_samples is not None
            else None
        )
        return (
            start_sample / sampling_rate,
            num_samples / sampling_rate if num_samples is not None else None,
        )


@dataclass
class Volume(AudioAugment):
    """
    Volume perturbation effect, the same one as invoked with `sox vol` in the command line.

    It changes the amplitude of the original samples, so the absolute values of output samples will
    be smaller or greater, depending on the vol factor.
    """

    factor: float

    def __call__(self, samples: np.ndarray, sampling_rate: int) -> np.ndarray:
        if isinstance(samples, np.ndarray):
            samples = torch.from_numpy(samples)
        augmented, _ = torchaudio.sox_effects.apply_effects_tensor(
            samples, sampling_rate, [["vol", str(self.factor)]]
        )
        return augmented.numpy()

    def reverse_timestamps(
        self,
        offset: Seconds,
        duration: Optional[Seconds],
        sampling_rate: Optional[int],  # Not used, made for compatibility purposes
    ) -> Tuple[Seconds, Optional[Seconds]]:
        """
        This method just returnes the original offset and duration as volume perturbation
        doesn't change any these audio properies.
        """

        return offset, duration

