import json
from inspect import isgenerator
import mimetypes
from pathlib import Path
from functools import partial
import numpy as np
from typing import Callable, Dict, Iterable, Optional, Union
import torch
import torchaudio.compliance.kaldi as kaldi


from uphill.core.utils import (
    Channels, Pathlike, Seconds,
    fastcopy,
    parallel_for
)
from uphill.core.audio import index_by_id_and_check
from uphill.core.text import Tokenizer, Vocabulary
from uphill.document import Document
from uphill import loggerx

from .mixins import AllMixins


class DocumentArray(AllMixins):
    doc_header = """
    :class:`~uphill.DocumentArray` represents a collection of documents, indexed by document IDs.
    It does not contain any annotation such as the transcript or the speaker identity --
    just the information needed to retrieve a document such as its path, URL, blob, tensor,
    and some document metadata.

    It also supports (de)serialization to/from YAML/JSON/etc. and takes care of mapping between
    rich Python classes and YAML/JSON/etc. primitives during conversion.
    """
    
    doc_body = """
    No need to pre-defined sub-class of `~uphill.DocumentArray` like `~uphill.AudioDocumentArray`, 
    it would automatically return matched sub-class of of `~uphill.DocumentArray` according to the 
    inputed documents, inputed dicts or file patterns.
    """
    
    doc_example = """
    Examples:

        :class:`~uphill.DocumentArray` can be created from an iterable of :class:`~uphill.Document` objects::

            >>> from uphill import Document, DocumentArray
            >>> audio_paths = ['123-5678.wav', ...]
            >>> da = DocumentArray.from_documents(Document.from_file(p) for p in audio_paths)

        As well as from a directory, which will be scanned recursively for files with parallel processing::

            >>> da_from_dir = DocumentArray.from_dir('data/audio', pattern='*.wav', num_jobs=4)

        It behaves similarly to a ``dict``::

            >>> '123-5678' in da
            True
            >>> doc = da['123-5678']
            >>> for doc in da:
            >>>    pass
            >>> len(da)
            127

        It also provides some utilities for I/O::

            >>> da.to_file('documents.jsonl')
            >>> da.to_file('documents.json.gz')  # auto-compression
            >>> da_from_jsonl = DocumentArray.from_file('documents.jsonl')

        Manipulation::

            >>> da_longer_than_5s = da.filter(lambda r: r.duration > 5)
            >>> da_first_100 = da.subset(first=100)
            >>> da_split_into_4 = da.split(num_splits=4)
            >>> da_shuffled = da.shuffle()

    """
    __doc__ = doc_header + doc_body + doc_example
    
    NAME_TO_DOCUMENTARRAY = {}
    DOCUMENTARRAY_TO_NAME = {}
    SAVE_MEMORY: bool = True
    
    def __init_subclass__(cls, **kwargs):
        if cls.__name__ not in DocumentArray.NAME_TO_DOCUMENTARRAY:
            key_name = cls.__name__.lower().replace("documentarray", "")
            DocumentArray.NAME_TO_DOCUMENTARRAY[key_name] = cls
            DocumentArray.DOCUMENTARRAY_TO_NAME[cls] = key_name
        super().__init_subclass__(**kwargs)
    
    def __init__(self, documents: Iterable[Document] = None) -> None:
        if documents is None:
            self.documents = {}
        elif isgenerator(documents):
            self.documents = index_by_id_and_check(documents)
        elif len(documents) > 0:
            if isinstance(documents[0], str):
                self.documents = {}
                for document_str in documents:
                    key = self.decompress_item(document_str).id
                    assert key not in self.documents, f"Duplicated manifest ID: {key}"
                    self.documents[key] = document_str
            else:
                self.documents = index_by_id_and_check(documents)
        else:
            self.documents = {}

    ################################
    ## property getter and setter ##
    ################################
    @property
    def data(self) -> Union[Dict[str, Document], Iterable[Document]]:
        """Alias property for ``self.documents``"""
        return self.documents

    @data.setter
    def data(self, key_value_dict: Dict[str, Document]):
        self.documents = key_value_dict

    @property
    def values(self) -> Union[Dict[str, Document], Iterable[Document]]:
        return self.documents.values()
    
    @property
    def ids(self) -> Iterable[str]:
        return self.documents.keys()


    ############################
    ## construction functions ##
    ############################
    @classmethod
    def from_documents(cls, documents: Iterable[Document]) -> "DocumentArray":
        if not isinstance(cls, DocumentArray):
            return cls(documents=documents)
        first_elem_name = next(iter(documents)).__class__.__name__.lower()
        DocumentArrayClass = DocumentArray
        if first_elem_name in DocumentArray.NAME_TO_DOCUMENTARRAY.keys():
            DocumentArrayClass = DocumentArray.NAME_TO_DOCUMENTARRAY[first_elem_name]
        return DocumentArrayClass(documents=documents)

    from_items = from_documents

    @staticmethod
    def from_dir(
        path: Pathlike,
        pattern: str,
        num_jobs: int = 1,
        document_id: Optional[Callable[[Path], str]] = None,
    ):
        """
        Recursively scan a directory ``path`` for files that match the given ``pattern`` and create
        a :class:`.DocumentArray` manifest for them.
        Suitable to use when each physical file represents a separate document session.

        .. caution::
            If a document session consists of multiple files (e.g. one per channel),
            it is advisable to create each :class:`.Document` object manually, with each
            file represented as a separate :class:`.DataSource` object, and then
            a :class:`DocumentArray` that contains all the documents.

        :param path: Path to a directory of audio of files (possibly with sub-directories).
        :param pattern: A bash-like pattern specifying allowed filenames, e.g. ``*.wav`` or ``session1-*.flac``.
        :param num_jobs: The number of parallel workers for reading audio files to get their metadata.
        :param document_id: A function which takes the audio file path and returns the document ID. If not
            specified, the filename will be used as the document ID.
        :return: a new ``DocumentArray`` instance pointing to the file.
        """
        mime_type_name = mimetypes.guess_type(pattern)[0].split("/")[0]

        file_read_worker = partial(
            Document.from_uri,
            id=document_id,
        )
        DocumentArrayClass = DocumentArray
        if mime_type_name in DocumentArray.NAME_TO_DOCUMENTARRAY:
            DocumentArrayClass = DocumentArray.NAME_TO_DOCUMENTARRAY[mime_type_name]

        loggerx.info(f"Scanning files ({path}/{pattern})")
        return DocumentArrayClass(
            documents=parallel_for(file_read_worker, Path(path).rglob(pattern), num_jobs=num_jobs)
        )

    @staticmethod
    def from_dict(data: Iterable[Union[dict, str]]) -> "DocumentArray":
        assert len(data) > 0
        if DocumentArray.SAVE_MEMORY:
            assert isinstance(data[0], str), "save memory, should be string"
            mime_type_name = json.loads(data[0])["__classname__"]
            DocumentArrayClass = DocumentArray
            if mime_type_name in DocumentArray.NAME_TO_DOCUMENTARRAY:
                DocumentArrayClass = DocumentArray.NAME_TO_DOCUMENTARRAY[mime_type_name]
            return DocumentArrayClass(
                documents=data
            )
        else:
            assert isinstance(data[0], dict), "should be dictionary"
            assert data[0]["__classname__"] in DocumentArray.NAME_TO_DOCUMENTARRAY
            mime_type_name = data[0]["__classname__"]
            DocumentArrayClass = DocumentArray
            if mime_type_name in DocumentArray.NAME_TO_DOCUMENTARRAY:
                DocumentArrayClass = DocumentArray.NAME_TO_DOCUMENTARRAY[mime_type_name]
            return DocumentArrayClass(
                documents=(Document.from_dict(raw_rec) for raw_rec in data)
            )

    ##################
    ### operations ###
    ##################
    def append(self, document: Union[Document, str], force_update: bool = False) -> None:
        if isinstance(document, str):
            document = self.decompress_item(document)
        if document.id in self.documents:
            if not force_update:
                loggerx.warning(f"Duplicated Document ID: {document.id}, not update")
                return
            loggerx.warning(f"Duplicated Document ID: {document.id}, update to new document")
        # compress document
        self.documents[document.id] = self.compress_item(document)
    
    def with_path_prefix(self, path: Pathlike) -> "DocumentArray":
        return  fastcopy(self, documents=(r.with_path_prefix(path) for r in self))


    ##########################
    ### internal functions ###
    ##########################
    def _item_from_dict(self, item: Dict):
        return Document.from_dict(item)
    
    def __eq__(self, other: "DocumentArray") -> bool:
        return self.documents == other.documents

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(len={len(self)})"

    def __contains__(self, item: Union[str, Document]) -> bool:
        key = item if isinstance(item, str) else item.id
        return key in self.documents

    def __getitem__(self, document_id_or_index: Union[int, str]) -> Document:
        if isinstance(document_id_or_index, int):
            if document_id_or_index >= len(self):
                raise IndexError(f"array index out of range, array length is {len(self)}")
            # ~100x faster than list(dict.values())[index] for 100k elements
            document = next(
                val 
                for idx, val in enumerate(self) 
                if idx == document_id_or_index
            )
        else:
            document = self.documents[document_id_or_index]
        return self.decompress_item(document)
    
    def __setitem__(self, document_id_or_index: Union[int, str], value):
        if isinstance(document_id_or_index, int):
            if document_id_or_index >= len(self):
                raise IndexError(f"array index out of range, array length is {len(self)}")
            # ~100x faster than list(dict.values())[index] for 100k elements
            document_id = self.decompress_item(
                next(
                    val 
                    for idx, val in enumerate(self) 
                    if idx == document_id_or_index
                )
            ).id
        else:
            document_id = document_id_or_index
        self.documents[document_id] = value

    def __iter__(self) -> Iterable[Document]:
        return iter(self.decompress_item(val) for val in self.values)

    def __len__(self) -> int:
        return len(self.documents)



class AudioDocumentArray(DocumentArray):
    doc_body = """
    When coming from Kaldi, think of it as ``wav.scp`` on steroids: :class:`~uphill.AudioDocumentArray`
    also has the information from *utt2dur* and *utt2num_samples*,
    is able to represent multi-channel documents and read a specified subset of channels,
    and support reading audio files directly, via a unix pipe, or downloading them on-the-fly from a URL
    (HTTPS/S3/Azure/GCP/etc.).
    """
    doc_example = """
        And lazy data augmentation/transformation, that requires to adjust some information
        in the manifest (e.g., ``num_samples`` or ``duration``).
        Note that in the following examples, the audio is untouched -- the operations are stored in the manifest,
        and executed upon reading the audio::

            >>> da_sp = da.perturb_speed(factor=1.1)
            >>> da_vp = da.perturb_volume(factor=2.)
            >>> da_24k = da.resample(24000)
    """
    __doc__ = DocumentArray.doc_header + doc_body + DocumentArray.doc_example + doc_example
    
    ################################
    ## property getter and setter ##
    ################################
    def num_channels(self, document_id: str) -> int:
        return self.documents[document_id].num_channels

    def sampling_rate(self, document_id: str) -> int:
        return self.documents[document_id].sampling_rate

    def num_samples(self, document_id: str) -> int:
        return self.documents[document_id].num_samples

    def duration(self, document_id: str) -> float:
        return self.documents[document_id].duration
    
    def load_audio(
        self,
        document_id: str,
        channels: Optional[Channels] = None,
        offset_seconds: float = 0.0,
        duration_seconds: Optional[float] = None,
    ) -> np.ndarray:
        return self.documents[document_id].load_audio(
            channels=channels, offset=offset_seconds, duration=duration_seconds
        )

    ##################
    ### operations ###
    ##################
    def perturb_speed(self, factor: float, affix_id: bool = True) -> "AudioDocumentArray":
        """
        Return a new ``AudioDocumentArray`` that will lazily perturb the speed while loading audio.
        The ``num_samples`` and ``duration`` fields are updated to reflect the
        shrinking/extending effect of speed.

        :param factor: The speed will be adjusted this many times (e.g. factor=1.1 means 1.1x faster).
        :param affix_id: When true, we will modify the ``AudioDocument.id`` field
            by affixing it with "_sp{factor}".
        :return: a ``AudioDocumentArray`` containing the perturbed ``AudioDocument`` objects.
        """
        return fastcopy(self, documents=(r.perturb_speed(factor=factor, affix_id=affix_id) for r in self))

    def perturb_tempo(self, factor: float, affix_id: bool = True) -> "AudioDocumentArray":
        """
        Return a new ``AudioDocumentArray`` that will lazily perturb the tempo while loading audio.
        The ``num_samples`` and ``duration`` fields are updated to reflect the
        shrinking/extending effect of tempo.

        :param factor: The speed will be adjusted this many times (e.g. factor=1.1 means 1.1x faster).
        :param affix_id: When true, we will modify the ``AudioDocument.id`` field
            by affixing it with "_tp{factor}".
        :return: a ``AudioDocumentArray`` containing the perturbed ``AudioDocument`` objects.
        """
        return fastcopy(self, documents=(r.perturb_tempo(factor=factor, affix_id=affix_id) for r in self))

    def perturb_volume(self, factor: float, affix_id: bool = True) -> "AudioDocumentArray":
        """
        Return a new ``AudioDocumentArray`` that will lazily perturb the volume while loading audio.

        :param factor: The volume scale to be applied (e.g. factor=1.1 means 1.1x louder).
        :param affix_id: When true, we will modify the ``AudioDocument.id`` field
            by affixing it with "_vp{factor}".
        :return: a ``AudioDocumentArray`` containing the perturbed ``AudioDocument`` objects.
        """
        return fastcopy(self, documents=(r.perturb_volume(factor=factor, affix_id=affix_id) for r in self))

    def resample(self, sampling_rate: int, affix_id: bool = True) -> "AudioDocumentArray":
        """
        Apply resampling to all documents in the ``AudioDocumentArray`` and return a new ``AudioDocumentArray``.
        :param sampling_rate: The new sampling rate.
        :return: a new ``AudioDocumentArray`` with lazily resampled ``AudioDocument`` objects.
        """
        return fastcopy(self, documents=(r.resample(sampling_rate, affix_id=affix_id) for r in self))

    def compute_cmvn(self, feat_dim: int=80, num_jobs: int=16) -> Dict:
        mean_stats = torch.zeros(feat_dim)
        var_stats = torch.zeros(feat_dim)
        num_frames = 0
        
        def _feature_fn(document_id):
            waveform = self.load_audio(document_id=document_id)
            sampling_rate = self.sampling_rate(document_id=document_id)
            
            waveform = waveform * (1 << 15)
            if len(waveform.shape) == 1:
                length = waveform.shape[0]
            else:
                length = waveform.shape[1]
            if length < 0.025 * sampling_rate:
                loggerx.warning(f"Document(id={document_id}) too short, only {length} samples")
                return None
            featform = kaldi.fbank(torch.Tensor(waveform),
                              num_mel_bins=feat_dim,
                              dither=0.0,
                              energy_floor=0.0,
                              sample_frequency=sampling_rate)
            return featform
        
        for featform in parallel_for(
            _feature_fn, self.documents.keys(), num_jobs=num_jobs
        ):
            if featform is None:
                continue
            mean_stats += torch.sum(featform, axis=0)
            var_stats += torch.sum(torch.square(featform), axis=0)
            num_frames += featform.shape[0]
        return {
            "num_frames": num_frames,
            "mean_stats": mean_stats,
            "var_stats": var_stats
        }



class TextDocumentArray(DocumentArray):
    def load_text(self, document_id):
        return self[document_id].load_text()
    
    def get_corpus(self):
        corpus = []
        for document_id in self.ids:
            text = self.load_text(document_id=document_id)
            corpus.append(text)
        return corpus
    
    def get_vocab(self, tokenizer: Tokenizer, num_jobs: int = 4) -> Vocabulary:
        
        def _tokenize_fn(document_id):
            text = self.load_text(document_id=document_id)
            tokens = tokenizer.tokenize(text)
            return tokens
        
        vocab = Vocabulary()
        for tokens in parallel_for(
            _tokenize_fn, self.ids, num_jobs=num_jobs
        ):
            if tokens is None or len(tokens) == 0:
                continue
            vocab.extend(tokens)
        return vocab
        


class ImageDocumentArray(DocumentArray):
    ...


class VideoDocumentArray(DocumentArray):
    ...


class AlignmentDocumentArray(DocumentArray):    
    def load_tensor(
        self, 
        document_id: str,
        window_size: float=0.01, 
        hop_size: float=0.025
    ) -> np.ndarray:
        return self.documents[document_id].load_tensor(
            window_size=window_size, hop_size=hop_size
        )


    ###############################
    ### augmentation operations ###
    ###############################
    def perturb_speed(self, factor: float, sampling_rate: int, affix_id: bool = True) -> "AlignmentDocumentArray":
        """
        Return a new ``AlignmentDocument`` that will lazily perturb the speed while loading audio.
        The ``num_samples`` and ``duration`` fields are updated to reflect the
        shrinking/extending effect of speed.

        :param factor: The speed will be adjusted this many times (e.g. factor=1.1 means 1.1x length).
        :param affix_id: When true, we will modify the ``AlignmentDocument.id`` field
            by affixing it with "_length{factor}_dim{dim}".
        :return: a modified copy of the current ``AlignmentDocument``.
        """
        return fastcopy(self, documents=(r.perturb_speed(factor=factor, affix_id=affix_id) for r in self))

    
    def with_offset(self, offset: Seconds, affix_id: bool = True) -> "AlignmentDocumentArray":
        """Return an identical ``AlignmentDocument``, but with the ``offset`` added to each source."""
        return fastcopy(self, documents=(r.with_offset(offset=offset, affix_id=affix_id) for r in self))


    def trim(self, end: Seconds, start: Seconds = 0, affix_id: bool = True) -> "AlignmentDocumentArray":
        """
        Return an identical ``AlignmentDocument``, but ensure that ``self.start`` is not negative (in which case
        it's set to 0) and ``self.end`` does not exceed the ``end`` parameter. If a `start` is optionally
        provided, the document is trimmed from the left (note that start should be relative to the cut times).
        """
        return fastcopy(self, documents=(r.trim(end=end, start=start, affix_id=affix_id) for r in self))


    def transform(self, transform_fn: Callable[[str], str], affix_id: bool = True) -> "AlignmentDocumentArray":
        """
        Perform specified transformation on the alignment content.
        """
        return fastcopy(self, documents=(r.transform(transform_fn=transform_fn, affix_id=affix_id) for r in self))

