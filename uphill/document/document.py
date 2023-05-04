import os
from typing_extensions import Literal
import numpy as np
from pathlib import Path
from decimal import ROUND_HALF_UP
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
)
from typeguard import check_argument_types


from uphill.core.utils import (
    ArrayType, Pathlike, Channels, Seconds,
    fastcopy, asdict_nonull
)
from uphill.core.audio import (
    AudioAugment, Speed, Volume, Resample, Tempo,
    torchaudio_info,
    compute_num_samples, perturb_num_samples, compute_num_windows,
    SetContainingAnything
)
from uphill import loggerx

from .mixins import AllMixin
from .source import DataSource, AlignmentDataSource


@dataclass(repr=False, eq=False)
class Document(AllMixin):
    
    sources: List[DataSource]
    id: str = field(default_factory=lambda: os.urandom(12).hex())
    transforms: Optional[List[Dict]] = None
    # Store anything else the user might want.
    custom: Optional[Dict[str, Any]] = None
    
    NAME_TO_DOCUMENT = {}
    DOCUMENT_TO_NAME = {}

    def __init_subclass__(cls, **kwargs):
        if cls.__name__ not in Document.NAME_TO_DOCUMENT:
            key_name = cls.__name__.lower().replace("document", "")
            Document.NAME_TO_DOCUMENT[key_name] = cls
            Document.DOCUMENT_TO_NAME[cls] = key_name
        super().__init_subclass__(**kwargs)
        
    def __post_init__(self):
        pass
    
    ################################
    ## property getter and setter ##
    ################################
    
    ############################
    ## construction functions ##
    ############################
    @staticmethod
    def from_uri(
        uri: Pathlike,
        id: Optional[Union[str, Callable[[Path], str]]] = None,
        *args, **kwargs
    ) -> "Document":
        """
        Read an audio file's header and create the corresponding ``Document``.
        Suitable to use when each physical file represents a separate utterance session.

        .. caution::
            If a utterance session consists of multiple files (e.g. one per channel),
            it is advisable to create the ``Document`` object manually, with each
            file represented as a separate ``DataSource`` object.

        :param uri: Path to an audio file supported by libsoundfile (pysoundfile).
        :param id: utterance id, when not specified ream the filename's stem ("x.wav" -> "x").
            It can be specified as a string or a function that takes the utterance path and returns a string.
        :return: a new ``Document`` instance pointing to the file.
        """
        id = (
            Path(uri).stem
            if id is None
            else id(uri)
            if callable(id)
            else id
        )
        source_id = Path(uri).stem
        data_source = DataSource.from_uri(id=source_id, uri=str(uri))
        DocumentClass = Document.NAME_TO_DOCUMENT[data_source.mime_type.split("/")[0]]
        return DocumentClass(id=id, sources=[data_source], *args, **kwargs)

    @classmethod
    def from_blob(
        cls,
        data: bytes,
        mime_type: Literal["audio", "video", "image", "text"] = None, 
        id: Optional[Union[str, Callable[[Path], str]]] = None,
        *args, **kwargs
    ) -> "Document":
        """
        Like :meth:`.Document.from_uri`, but creates a manifest for a byte string with
        raw encoded audio data. This data is first decoded to obtain info such as the
        sampling rate, number of channels, etc. Then, the binary data is attached to the
        manifest. Calling :meth:`.Document.load_blob` does not perform any I/O and
        instead decodes the byte string contents in memory.

        .. note:: Intended use of this method is for packing Documents into archives
            where metadata and data should be available together
            (e.g., in WebDataset style tarballs).

        .. caution:: Manifest created with this method cannot be stored as JSON
            because JSON doesn't allow serializing binary data.

        :param data: bytes, byte string containing encoded audio contents.
        :param mime_type: str, MIME type of the byte string data.
        :param id: utterance id, unique string identifier.
        :return: a new ``Document`` instance that owns the byte string data.
        """
        assert check_argument_types()
        if cls == Document:
            # Default Call, need to decide which sub-class to be initialized
            assert mime_type is not None, f"mime_type is unkown, valid for construction"
            DocumentClass = Document.NAME_TO_DOCUMENT[mime_type]
        else:
            mime_type = Document.DOCUMENT_TO_NAME[cls]
            if mime_type == "alignment":
                mime_type = "tensor"
            DocumentClass = cls
        data_source = DataSource.from_blob(
            blob=data, 
            mime_type=mime_type, 
            id=id, 
            *args, **kwargs
        )
        DocumentClass = Document.NAME_TO_DOCUMENT[mime_type]
        document = DocumentClass(sources=[data_source])
        if id is not None:
            document.id = str(id)
        return document

    @classmethod
    def from_tensor(
        cls,
        data: ArrayType,
        mime_type: Literal["audio", "video", "image", "text"] = None, 
        id: Optional[Union[str, Callable[[Path], str]]] = None,
        *args, **kwargs
    ) -> "Document":
        """
        Like :meth:`.Document.from_uri`, but creates a manifest for a tensor. 
        This data is first decoded to obtain info such as the
        sampling rate, number of channels, etc. Then, the binary data is attached to the
        manifest. Calling :meth:`.Document.load_tensor` does not perform any I/O and
        instead decodes the byte string contents in memory.

        .. note:: Intended use of this method is for packing Documents into archives
            where metadata and data should be available together
            (e.g., in WebDataset style tarballs).

        :param data: ArrayType, tensor containing encoded audio contents.
        :param mime_type: str, MIME type of the tensor data.
        :param id: utterance id, unique string identifier.
        :return: a new ``Document`` instance that owns the tensor data.
        """
        assert check_argument_types()
        if cls == Document:
            # Default Call, need to decide which sub-class to be initialized
            assert mime_type in Document.NAME_TO_DOCUMENT, f"mime_type({mime_type}) invalid for construction"
            DocumentClass = Document.NAME_TO_DOCUMENT[mime_type]
        else:
            mime_type = Document.DOCUMENT_TO_NAME[cls]
            DocumentClass = cls
        data_source = DataSource.from_tensor(
            tensor=data, 
            mime_type=mime_type, 
            id=id, 
            *args, **kwargs
        )
        document = DocumentClass(sources=[data_source])
        if id is not None:
            document.id = str(id)
        return document

    ## this initial function only for `TextDocument`
    @staticmethod
    def from_text(
        text: str, 
        id: str = None, 
        *args, **kwargs
    ):
        source = DataSource.from_text(text=text, id=id)
        DocumentClass = Document.NAME_TO_DOCUMENT["text"]
        instance = DocumentClass(
            sources=[source], 
            *args, **kwargs)
        if id is not None:
            instance.id = str(id)
        return instance


    @staticmethod
    def from_dict(data: dict) -> "Document":
        DocumentClass = Document
        if "__classname__" in data:
            DocumentClass = Document.NAME_TO_DOCUMENT[data.pop("__classname__")]
        raw_sources = data.pop("sources")
        return DocumentClass(
            sources=[DataSource.from_dict(s) for s in raw_sources], **data
        )


    ###########################
    ### exporting functions ###
    ###########################
    def to_dict(self) -> dict:
        doc_info = asdict_nonull(self)
        doc_info['sources'] = [s.to_dict() for s in self.sources]
        doc_info["__classname__"] = self.DOCUMENT_TO_NAME[self.__class__]
        return doc_info


    ##################
    ### operations ###
    ##################
    def with_path_prefix(self, path: Pathlike) -> "Document":
        return fastcopy(self, sources=[s.with_path_prefix(path) for s in self.sources])
    
    def add_source(self, source: DataSource) -> None:
        self.sources.append(source)


    ##########################
    ### internal functions ###
    ##########################
    def __eq__(self, other: 'Document') -> bool:
        if self.id != other.id:
            return False 
        if self.sources.sort() != other.sources.sort():
            return False
        if (self.transforms is not None and other.transforms is None) or \
            (self.transforms is None and other.transforms is not None):
            return False
        if self.transforms is not None and other.transforms is not None and len(self.transforms) != len(other.transforms):
            return False
        # TODO: check transforms content
        return True


@dataclass(repr=False, eq=False)
class AudioDocument(Document):
    ## for Audio document
    num_channels: List[int] = field(default_factory=list)
    sampling_rate: int = None
    num_samples: int = None
    start: float = 0.0
    duration: float = None
    
    def __post_init__(self):
        super().__post_init__()
        if self.sampling_rate is not None or self.num_samples is not None:
            return
        uri = self.sources[0].uri
        if not os.path.exists(uri):
            loggerx.warning(f"Uri not exist: {uri}")
            return
        audio_info = torchaudio_info(uri)
        self.sampling_rate = audio_info.samplerate
        self.num_channels = list(range(audio_info.channels))
        
        if self.start > 0.0 or self.duration is not None:
            # num samples cannot copied from audio file
            if self.duration is None:
                self.duration = audio_info.duration
            else:
                self.duration = min(self.duration, audio_info.duration)
            self.num_samples = compute_num_samples(
                duration=self.duration,
                sampling_rate=self.sampling_rate,
            )
        else:
            # copied from audio file
            self.num_samples = audio_info.frames
            self.duration = audio_info.duration
        
        ## TODO:
        self.sources[0].num_channels = self.num_channels
    
    def load_audio(
        self,
        channels: Optional[Channels] = None,
        offset: float = 0.0,
        duration: Optional[float] = None,
    ) -> np.ndarray:
        """
        Read the audio samples from the underlying audio source (path, URL, unix pipe/command).

        :param channels: int or iterable of ints, a subset of channel IDs to read (reads all by default).
        :param offset: float, where to start reading the audio (at offset 0 by default).
            Note that it is only efficient for local filesystem files, i.e. URLs and commands will read
            all the samples first and discard the unneeded ones afterwards.
        :param duration: float, indicates the total audio time to read (starting from ``offset``).
        :return: a numpy array of audio samples with shape ``(num_channels, num_samples)``.
        """
        assert offset <= self.duration, (
            f"Cannot load audio because the Document's duration {self.duration}s "
            f"is smaller than the requested offset {offset}s."
        )

        if channels is None:
            channels = SetContainingAnything()
        else:
            channels = frozenset([channels] if isinstance(channels, int) else channels)
            channel_ids = sorted(cid for source in self.sources for cid in source.num_channels)
            utterance_channels = frozenset(channel_ids)
            assert channels.issubset(utterance_channels), (
                "Requested to load audio from a channel "
                "that does not exist in the utterance: "
                f"(utterance channels: {utterance_channels} -- "
                f"requested channels: {channels})"
            )
        transforms = [
            AudioAugment.from_dict(params) for params in self.transforms or []
        ]

        # to ensure not exceed the original left boundary
        offset_aug = max(offset + self.start, self.start)
        if duration is None:
            duration_aug = self.duration
        else:
            duration_aug = duration if self.duration is None else self.duration + duration
        # to ensure not exceed the original right boundary
        duration_aug = min(self.start+self.duration-offset_aug, duration_aug)
        # Do a "backward pass" over data augmentation transforms to get the
        # offset and duration for loading a piece of the original audio.
        for tfn in reversed(transforms):
            offset_aug, duration_aug = tfn.reverse_timestamps(
                offset=offset_aug,
                duration=duration_aug,
                sampling_rate=self.sampling_rate,
            )

        samples_per_source = []
        for source in self.sources:
            # Case: source not requested
            if not channels.intersection(source.num_channels):
                continue
            samples = source.load_tensor(
                offset=offset_aug,
                duration=duration_aug
            )

            # Case: two-channel audio file but only one channel requested
            #       it might not be optimal to load all channels, but IDK if there's anything we can do about it
            channels_to_remove = [
                idx for idx, cid in enumerate(source.num_channels) if cid not in channels
            ]
            if channels_to_remove:
                samples = np.delete(samples, channels_to_remove, axis=0)
            samples_per_source.append(samples)

        # Stack all the samples from all the sources into a single array.
        audio = self._stack_audio_channels(samples_per_source)

        # We'll apply the transforms now (if any).
        for tfn in transforms:
            audio = tfn(audio, self.sampling_rate)

        # Transformation chains can introduce small mismatches in the number of samples:
        # we'll fix them here, or raise an error if they exceeded a tolerance threshold.
        # audio = assert_and_maybe_fix_num_samples(
        #     audio, offset=offset, duration=duration, utterance=self
        # )
        return audio
    
    ###############################
    ### augmentation operations ###
    ###############################
    def perturb_speed(self, factor: float, affix_id: bool = True) -> "AudioDocument":
        """
        Return a new ``AudioDocument`` that will lazily perturb the speed while loading audio.
        The ``num_samples`` and ``duration`` fields are updated to reflect the
        shrinking/extending effect of speed.

        :param factor: The speed will be adjusted this many times (e.g. factor=1.1 means 1.1x faster).
        :param affix_id: When true, we will modify the ``AudioDocument.id`` field
            by affixing it with "_sp{factor}".
        :return: a modified copy of the current ``AudioDocument``.
        """
        transforms = self.transforms.copy() if self.transforms is not None else []
        transforms.append(Speed(factor=factor).to_dict())
        num_samples_after_perturb = perturb_num_samples(self.num_samples, factor)
        duration_after_perturb = num_samples_after_perturb / self.sampling_rate
        
        return fastcopy(
            self,
            id=f"{self.id}_sp{factor}" if affix_id else self.id,
            transforms=transforms,
            num_samples=num_samples_after_perturb,
            duration=duration_after_perturb,
        )

    def perturb_tempo(self, factor: float, affix_id: bool = True) -> "AudioDocument":
        """
        Return a new ``AudioDocument`` that will lazily perturb the tempo while loading audio.

        Compared to speed perturbation, tempo preserves pitch.
        The ``num_samples`` and ``duration`` fields are updated to reflect the
        shrinking/extending effect of tempo.

        :param factor: The tempo will be adjusted this many times (e.g. factor=1.1 means 1.1x faster).
        :param affix_id: When true, we will modify the ``AudioDocument.id`` field
            by affixing it with "_tp{factor}".
        :return: a modified copy of the current ``AudioDocument``.
        """
        transforms = self.transforms.copy() if self.transforms is not None else []
        transforms.append(Tempo(factor=factor).to_dict())
        num_samples_after_perturb = perturb_num_samples(self.num_samples, factor)
        duration_after_perturb = num_samples_after_perturb / self.sampling_rate
        return fastcopy(
            self,
            id=f"{self.id}_tp{factor}" if affix_id else self.id,
            transforms=transforms,
            num_samples=num_samples_after_perturb,
            duration=duration_after_perturb,
            
        )

    def perturb_volume(self, factor: float, affix_id: bool = True) -> "AudioDocument":
        """
        Return a new ``AudioDocument`` that will lazily perturb the volume while loading audio.

        :param factor: The volume scale to be applied (e.g. factor=1.1 means 1.1x louder).
        :param affix_id: When true, we will modify the ``AudioDocument.id`` field
            by affixing it with "_tp{factor}".
        :return: a modified copy of the current ``AudioDocument``.
        """
        transforms = self.transforms.copy() if self.transforms is not None else []
        transforms.append(Volume(factor=factor).to_dict())
        return fastcopy(
            self,
            id=f"{self.id}_vp{factor}" if affix_id else self.id,
            transforms=transforms,
        )

    def resample(self, sampling_rate: int, affix_id: bool = True) -> "AudioDocument":
        """
        Return a new ``AudioDocument`` that will be lazily resampled while loading audio.
        :param sampling_rate: The new sampling rate.
        :return: A resampled ``AudioDocument``.
        """
        if sampling_rate == self.sampling_rate:
            return fastcopy(self)

        transforms = self.transforms.copy() if self.transforms is not None else []

        '''
        # TODO: OPUS is a special case for resampling.
        `not any(str(s.uri).endswith(".opus") for s in self.sources)`
        # Normally, we use Torchaudio SoX bindings for resampling,
        # but in case of OPUS we ask FFMPEG to resample it during
        # decoding as its faster.
        # Because of that, we have to skip adding a transform
        # for OPUS files and only update the metadata in the manifest.
        '''
        transforms.append(
            Resample(
                source_sampling_rate=self.sampling_rate,
                target_sampling_rate=sampling_rate,
            ).to_dict()
        )

        num_samples_after_resample = compute_num_samples(
            self.duration, sampling_rate, rounding=ROUND_HALF_UP
        )
        # Duration might need an adjustment when doing a non-trivial resampling
        # (e.g. 16000 -> 22050), where the resulting number of samples cannot
        # correspond to old duration exactly.
        duration_after_resample = num_samples_after_resample / sampling_rate
        return fastcopy(
            self,
            id=f"{self.id}_rs{sampling_rate}" if affix_id else self.id,
            transforms=transforms,
            sampling_rate=sampling_rate,
            num_samples=num_samples_after_resample,
            duration=duration_after_resample,
        )

    def _stack_audio_channels(self, samples_per_source: List[np.ndarray]) -> np.ndarray:
        # There may be a mismatch in the number of samples between different channels. We
        # check if the mismatch is within a reasonable tolerance and if so, we pad
        # all channels to the length of the longest one.
        allowed_diff = int(
            compute_num_samples(
                0.025,
                sampling_rate=self.sampling_rate,
            )
        )
        if len(samples_per_source) > 1:
            # Make all arrays 2D
            samples_per_source = [
                s[None, :] if s.ndim == 1 else s for s in samples_per_source
            ]
            max_samples = max(s.shape[1] for s in samples_per_source)
            for s in samples_per_source:
                if max_samples - s.shape[1] <= allowed_diff:
                    s = np.pad(s, ((0, 0), (0, max_samples - s.shape[1])), "constant")
                else:
                    raise ValueError(
                        f"The mismatch between the number of samples in the "
                        f"different channels of the utterance {self.id} is "
                        f"greater than the allowed tolerance {0.025}."
                    )
            audio = np.concatenate(samples_per_source, axis=0)
        else:
            # shape: (n_channels, n_samples)
            audio = np.vstack(samples_per_source)
        return audio


@dataclass(repr=False, eq=False)
class TextDocument(Document):
    def load_text(self, delimiter: str = " <SEP> ") -> str:
        contents = []
        for source in self.sources:
            contents.append(source.load_text())
        return delimiter.join(contents)


@dataclass(repr=False, eq=False)
class VideoDocument(Document):
    ...


@dataclass(repr=False, eq=False)
class ImageDocument(Document):
    ...
    
    
@dataclass(repr=False, eq=False)
class AlignmentDocument(Document):
    # sources: List[DataSource]
    
    @staticmethod
    def from_segments(
        segments: Iterable[Tuple[Union[str, int], float, float]],
        id: Optional[Union[str, Callable[[Path], str]]] = None,
    ) -> "AlignmentDocument":
        assert check_argument_types()
        document = AlignmentDocument(
            sources=[ 
                AlignmentDataSource(symbol=segment[0], start=segment[1], duration=segment[2]) 
                for segment in segments
            ]
        )
        if id is not None:
            document.id = str(id)
        return document


    def load_tensor(self, window_size: float=0.025, frame_shift: float=0.01) -> ArrayType:
        labels = []
        absolute_start = None
        for idx, source in enumerate(self.sources):
            exist_num_labels = len(labels)
            segment_start, segment_end = source.start, source.end
            if idx == 0:
                ## first segment to get the absolute start of whole segment
                absolute_start = segment_start
            total_num_labels_til_now = compute_num_windows(
                duration = segment_end - absolute_start,
                window_size = window_size,
                frame_shift = frame_shift
            )
            current_num_labels = total_num_labels_til_now - exist_num_labels
            labels.extend([source.symbol] * current_num_labels)
        return np.array(labels, dtype=np.int32)


    ###############################
    ### augmentation operations ###
    ###############################
    def perturb_speed(self, factor: float, sampling_rate: int, affix_id: bool = True) -> "AlignmentDocument":
        """
        Return a new ``AlignmentDocument`` that will lazily perturb the speed while loading audio.
        The ``num_samples`` and ``duration`` fields are updated to reflect the
        shrinking/extending effect of speed.

        :param factor: The speed will be adjusted this many times (e.g. factor=1.1 means 1.1x length).
        :param affix_id: When true, we will modify the ``AlignmentDocument.id`` field
            by affixing it with "_length{factor}_dim{dim}".
        :return: a modified copy of the current ``AlignmentDocument``.
        """
        return fastcopy(
            self,
            id = f"{self.id}_length{factor}" if affix_id else self.id,
            sources=[
                source.perturb_speed(factor=factor, sampling_rate=sampling_rate)
                for source in self.sources
            ]
        )
    
    def with_offset(self, offset: Seconds, affix_id: bool = True) -> "AlignmentDocument":
        """Return an identical ``AlignmentDocument``, but with the ``offset`` added to each source."""
        return fastcopy(
            self,
            id = f"{self.id}_offset{offset}" if affix_id else self.id,
            sources=[
                source.with_offset(offset=offset)
                for source in self.sources
            ]
        )

    def trim(self, end: Seconds, start: Seconds = 0, affix_id: bool = True) -> "AlignmentDocument":
        """
        Return an identical ``AlignmentDocument``, but ensure that ``self.start`` is not negative (in which case
        it's set to 0) and ``self.end`` does not exceed the ``end`` parameter. If a `start` is optionally
        provided, the document is trimmed from the left (note that start should be relative to the cut times).
        """
        sources=[]
        for source in self.sources:
            trim_source = source.trim(end=end, start=start)
            if trim_source is None:
                continue
            sources.append(trim_source)
        return fastcopy(
            self,
            id = f"{self.id}_trim{end}" if affix_id else self.id,
            sources=sources
        )

    def transform(self, transform_fn: Callable[[str], str], affix_id: bool = True) -> "AlignmentDocument":
        """
        Perform specified transformation on the alignment content.
        """
        return fastcopy(
            self,
            id = f"{self.id}_transform-{id(transform_fn)}" if affix_id else self.id,
            sources=[
                source.transform(transform_fn=transform_fn)
                for source in self.sources
            ]
        )


