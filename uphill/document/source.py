import os
import warnings
import numpy as np
import mimetypes
import urllib
from io import BytesIO
from collections import Counter
from pathlib import Path
import torch
import torchaudio
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Union,
    Callable,
)
from typing_extensions import Literal


from uphill.errors import InitClassError
from uphill.core.audio import (
    read_audio,
    compute_num_samples, perturb_num_samples, 
)
from uphill.core.text import (
    Tokenizer, Vocabulary
)
from uphill.core.utils import (
    Pathlike, ArrayType, Seconds,
    SmartOpen, fastcopy, asdict_nonull
)
from uphill import loggerx

from .mixins import AllMixin
from .helper import _uri_to_blob, _to_datauri


@dataclass(repr=False, eq=False)
class DataSource(AllMixin):
    """
    DataSource represents audio data that can be retrieved from somewhere.
    Supported sources of data are currently:
    - 'file' (formats supported by soundfile, possibly multi-channel)
    - 'url' (any URL type that is supported by "smart_open" library, e.g. http/https/s3/gcp/azure/etc.)
    - 'blob' (any format, read from a binary string attached to 'source' member of DataSource)
    - 'tensor' (any tensor format)
    """
    
    id: str = field(default_factory=lambda: os.urandom(12).hex())
    uri: Optional[Pathlike] = None
    _blob: Optional[bytes] = None
    _tensor: Optional[ArrayType] = None
    mime_type: Optional[str] = None
    # Store anything else the user might want.
    custom: Optional[Dict[str, Any]] = None
    
    NAME_TO_DATASOURCE = {}
    DATASOURCE_TO_NAME = {}
    
    def __init_subclass__(cls, **kwargs):
        if cls.__name__ not in DataSource.NAME_TO_DATASOURCE:
            key_name = cls.__name__.lower().replace("datasource", "")
            DataSource.NAME_TO_DATASOURCE[key_name] = cls
            DataSource.DATASOURCE_TO_NAME[cls] = key_name
        super().__init_subclass__(**kwargs)
    
    def __post_init__(self):
        key_name = self.__class__.__name__.lower().replace("datasource", "")
        if key_name != "text" and self.uri is None and self._blob is None and self._tensor is None:
            raise InitClassError("mime_type in ['uri', 'blob', 'tensor'] should at least one non NoneType")
        if self.uri is not None and not self.mime_type:
            self.uri = str(self.uri)
            mime_type = mimetypes.guess_type(self.uri)[0]
            if mime_type:
                self.mime_type = mime_type


    ################################
    ## property getter and setter ##
    ################################
    # @property
    # def uri(self):
    #     if self._uri is None:
    #         self._get_uri()
    #     return self._uri

    # @uri.setter
    # def uri(self, value):
    #     self._uri = value

    @property
    def tensor(self):
        if self._tensor is None:
            self._tensor = self.load_tensor()
        return self._tensor
    
    def load_tensor(self) -> ArrayType:
        raise NotImplementedError(f"load_tensor not implemented in {self.__class__}")
    
    @tensor.setter
    def tensor(self, value: ArrayType):
        self._tensor = value
    
    @property
    def blob(self):
        if self._blob is None:
            self._blob = self.load_blob()
        return self._blob

    def load_blob(self) -> bytes:
        raise NotImplementedError(f"load_blob not implemented in {self.__class__}")
    
    @blob.setter
    def blob(self, value: bytes):
        self._blob = value

    ############################
    ## construction functions ##
    ############################
    @staticmethod
    def from_uri(uri: Pathlike, id: str = None, *args, **kwargs) -> 'DataSource':
        main_mime_type = mimetypes.guess_type(str(uri))[0].split("/")[0]
        DataSourceClass = DataSource.NAME_TO_DATASOURCE[main_mime_type]
        instance = DataSourceClass(uri=uri, *args, **kwargs)
        if id is not None:
            instance.id = str(id)
        return instance
    
    ## TODO:
    @staticmethod
    def from_blob(
        blob: bytes, 
        mime_type: Literal["audio", "video", "image", "text"], 
        id: str = None,
        *args, **kwargs
    ) -> 'DataSource':
        assert mime_type in DataSource.NAME_TO_DATASOURCE, \
            f"mime_type only support {list(DataSource.NAME_TO_DATASOURCE.keys())}, "\
            f"not support {mime_type}"
        DataSourceClass = DataSource.NAME_TO_DATASOURCE[mime_type]
        instance = DataSourceClass(_blob=blob, mime_type=mime_type, *args, **kwargs)
        if id is not None:
            instance.id = str(id)
        return instance

    @staticmethod
    def from_tensor(
        tensor: ArrayType, 
        mime_type: Literal["audio", "video", "image", "text"], 
        id: str = None, 
        *args, **kwargs
    ) -> 'DataSource':
        assert mime_type in DataSource.NAME_TO_DATASOURCE, \
            f"mime_type only support {list(DataSource.NAME_TO_DATASOURCE.keys())}, "\
            f"not support {mime_type}"
        DataSourceClass = DataSource.NAME_TO_DATASOURCE[mime_type]
        instance = DataSourceClass(_tensor=tensor, mime_type=mime_type, *args, **kwargs)
        if id is not None:
            instance.id = str(id)
        return instance
    
    ## this initial function only for `TextDataSource`
    @staticmethod
    def from_text(
        text: str, 
        id: str = None, 
        *args, **kwargs
    ):
        DataSourceClass = DataSource.NAME_TO_DATASOURCE["text"]
        instance = DataSourceClass(text=text, mime_type="text/plain", *args, **kwargs)
        if id is not None:
            instance.id = str(id)
        return instance

    @staticmethod
    def from_dict(data: dict) -> "DataSource":
        DataSourceClass = DataSource
        if "__classname__" in data:
            DataSourceClass = DataSource.NAME_TO_DATASOURCE[data.pop("__classname__")]
        return DataSourceClass(**data)

    
    ###########################
    ### exporting functions ###
    ###########################
    def to_dict(self) -> dict:
        doc_info = asdict_nonull(self)
        doc_info["__classname__"] = self.DATASOURCE_TO_NAME[self.__class__]
        return doc_info


    ##################
    ### operations ###
    ##################
    def with_path_prefix(self, path: Pathlike) -> "DataSource":
        return fastcopy(self, uri=str(Path(path) / self.uri))


    ##########################
    ### internal functions ###
    ##########################
    def __eq__(self, other: 'DataSource') -> bool:
        if self.id != other.id or str(self.uri) != str(other.uri) or self.mime_type != other.mime_type:
            return False 
        if self._blob and other._blob and len(self._blob) != len(other._blob):
            return False
        if self._tensor is not None and other._tensor is not None and self._tensor.shape != other._tensor.shape:
            return False
        return True


@dataclass(repr=False, eq=False)
class AudioDataSource(DataSource):
    
    sampling_rate: int = None
    num_channels: List[int] = field(default_factory=list)
    
    def __post_init__(self):
        super().__post_init__()
        if self._tensor is not None:
            self._update_num_channels()
                
    def _update_num_channels(self):
        if len(self._tensor.shape) == 1:
            self.num_channels = [0]
        else:
            self.num_channels = list(range(self._tensor.shape[0]))

    def load_blob(self) -> bytes:
        blob = b""
        if self._blob is not None:
            blob = self._blob
        if self.uri is not None:
            blob = _uri_to_blob(str(self.uri))
        elif self._tensor is not None:
            blob = self.load_tensor_to_blob()
        else:
            loggerx.warning(f"load blob data failed")
        return blob

    def load_tensor_to_blob(self, format: Optional[str] = "wav") -> bytes:
        stream = BytesIO()
        tensor = self._tensor
        if isinstance(self._tensor, np.ndarray):
            tensor = torch.from_numpy(self._tensor)
        torchaudio.save(
            stream, tensor, self.sampling_rate, format=format, bits_per_sample=16
        )
        self._blob = stream.getvalue()
    
    def load_tensor(
        self,
        offset: float = 0.0,
        duration: Optional[float] = None,
    ) -> ArrayType:
        tensor = None
        if self._tensor is not None and offset == 0.0 and duration is None:
            tensor = self._tensor
        elif self.uri is not None:
            tensor = self.load_uri_to_tensor(offset=offset, duration=duration)
        elif self._blob is not None:
            tensor = self.load_blob_to_tensor(offset=offset, duration=duration)
        else:
            loggerx.warning(f"load tensor failed")
        return tensor
    
    def load_uri_to_tensor(
        self, 
        offset: float = 0.0,
        duration: Optional[float] = None
    ) -> ArrayType:
        """
        Load the DataSource (from files or URLs) with torchaudio,
        accounting for many audio formats and multi-channel inputs.
        Returns numpy array with shapes: (n_samples,) for single-channel,
        (n_channels, n_samples) for multi-channel.

        Note: The elements in the returned array are in the range [-1.0, 1.0]
        and are of dtype `np.float32`.
        """
        if urllib.parse.urlparse(str(self.uri)) in ['data', 'http', 'https']:
            if offset != 0.0 or duration is not None:
                warnings.warn(
                    "You requested a subset of a utterance that is read from URL. "
                    "Expect large I/O overhead if you are going to read many chunks like these, "
                    "since every time we will download the whole file rather than its subset."
                )
            with SmartOpen.open(self.uri, "rb") as f:
                source = BytesIO(f.read())
                samples, sampling_rate = read_audio(
                    source, offset=offset, duration=duration
                )
        else:
            samples, sampling_rate = read_audio(
                self.uri,
                offset=offset,
                duration=duration,
            )

        # explicit sanity check for duration as soundfile does not complain here
        if duration is not None:
            num_samples = (
                samples.shape[0] if len(samples.shape) == 1 else samples.shape[1]
            )
            available_duration = num_samples / sampling_rate
            if (
                available_duration < duration - 0.025
            ):  # set the allowance as 1ms to avoid float error
                raise Exception(
                    f"Requested more audio ({duration}s) than available ({available_duration}s)"
                )
        
        if len(samples.shape) == 1:
            self.num_channels = [0]
        else:
            self.num_channels = list(range(samples.shape[0]))
        return samples.astype(np.float32)
    
    def load_blob_to_tensor(
        self, 
        offset: float = 0.0,
        duration: Optional[float] = None
    ) -> ArrayType:
        source = BytesIO(self._blob)
        samples, sampling_rate = read_audio(
            source, offset=offset, duration=duration
        )
        # explicit sanity check for duration as soundfile does not complain here
        if duration is not None:
            num_samples = (
                samples.shape[0] if len(samples.shape) == 1 else samples.shape[1]
            )
            available_duration = num_samples / sampling_rate
            if (
                available_duration < duration - 0.025
            ):  # set the allowance as 1ms to avoid float error
                raise Exception(
                    f"Requested more audio ({duration}s) than available ({available_duration}s)"
                )
        if len(samples.shape) == 1:
            self.num_channels = [0]
        else:
            self.num_channels = list(range(samples.shape[0]))
        return samples.astype(np.float32)


@dataclass(repr=False, eq=False)
class TextDataSource(DataSource):

    text: Optional[str] = None

    def load_blob(self) -> bytes:
        self._blob = _uri_to_blob(self.uri)

    def load_tensor(self) -> ArrayType:
        raise NotImplementedError(f"Function[load_tensor] not implemented in {self.__class__}")

    def load_text(self, charset: str = 'utf-8') -> str:
        """load text automatically

        :param charset: charset may be any character set registered with IANA
        """
        text = None
        if self.text is not None:
            text = self.text
        elif self._blob is not None:
            text = self._blob.decode(charset)
        elif self.uri is not None:
            blob = _uri_to_blob(self.uri)
            text = blob.decode(charset)
        else:
            loggerx.warning(f"uri and blob are NoneType, load text failed.")
        return text

    def get_vocab(self, tokenizer: Tokenizer) -> Vocabulary:
        """Get the text vocabulary in a counter dict that maps from the word to 
            its frequency from all :attr:`text_fields`.

        :param tokenizer: a `Tokenizer` object that tokenizes a text and 
            detokenizes token sequence.
            
        :return: a `Vocabulary` object 
        """
        all_tokens = Counter()
        all_tokens.update(tokenizer.tokenize(self.text))
        # 0 for padding, 1 for unknown
        vocab  = {"<pad>": 0, "<unk>": 1}
        idx = len(vocab)
        for token, count in all_tokens.items():
            vocab[token] = idx
            idx += 1
        return Vocabulary.from_dict(vocab)

    def convert_text_to_tensor(
        self, tokenizer: Tokenizer, vocab: Vocabulary, dtype: str = 'int64',
    ):
        """Convert :attr:`.text` to :attr:`.tensor`.

        To get the vocab of a text document array, you can use 
        `uphill.TextDocumentArray.get_vocab` to get the vocab of a text array.
        
        :param tokenizer: a `Tokenizer` object that tokenizes a text and 
            detokenizes token sequence.
        :param vocab: a `Vocabulary` object that maps a word to an integer index 
            and inversely, 
        :param dtype: the dtype of the generated :attr:`.tensor`
        
        :return: 1D array
        """
        tokens = tokenizer.tokenize(self.text)
        tensor = [vocab[token] if vocab[token] > 0 else 0 for token in tokens]
        return np.array(tensor, dtype=dtype)

    def convert_tensor_to_text(
        self, tokenizer: Tokenizer, vocab: Vocabulary, 
    ):
        """Convert :attr:`.tensor` to :attr:`.text`.

        :param tokenizer: a `Tokenizer` object that tokenizes a text and 
            detokenizes token sequence.
        :param vocab: a `Vocabulary` object that maps a word to an integer index 
            and inversely, 
        
        :return: text
        """
        tokens = [vocab[v] for v in self.tensor]
        text = tokenizer.detokenize(tokens=tokens)
        return text

    def convert_text_to_datauri(
        self, charset: str = 'utf-8', base64: bool = False
    ):
        """Convert :attr:`.text` to data :attr:`.uri`.

        :param charset: charset may be any character set registered with IANA
        :param base64: used to encode arbitrary octet sequences into a form 
            that satisfies the rules of 7bit. Designed to be efficient for 
            non-text 8 bit and binary data. Sometimes used for text data that 
            frequently uses non-US-ASCII characters.

        :return: data uri
        """
        self.uri = _to_datauri(self.mime_type, self.text, charset, base64, binary=False)


@dataclass(repr=False, eq=False)
class ImageDataSource(DataSource):
    def load_blob(self) -> bytes:
        self._blob = _uri_to_blob(self.uri)
        
    def load_tensor(self) -> ArrayType:
        raise NotImplementedError(f"Function[load_tensor] not implemented in {self.__class__}")


@dataclass(repr=False, eq=False)
class VideoDataSource(DataSource):
    def load_blob(self) -> bytes:
        self._blob = _uri_to_blob(self.uri)
        
    def load_tensor(self) -> ArrayType:
        raise NotImplementedError(f"Function[load_tensor] not implemented in {self.__class__}")


@dataclass(repr=False, eq=False)
class AlignmentDataSource(DataSource):
    """
    This class contains an alignment item, for example a word, along with its
    start time (w.r.t. the start of recording) and duration. It can potentially
    be used to store other kinds of alignment items, such as subwords, pdfid's etc.

    We use dataclasses instead of namedtuples (even though they are potentially slower)
    because of a serialization bug in nested namedtuples and dataclasses in Python 3.7
    (see this: https://alexdelorenzo.dev/programming/2018/08/09/bug-in-dataclass.html).
    We can revert to namedtuples if we bump up the Python requirement to 3.8+.
    """

    symbol: Union[str, int] = None
    start: Seconds = None
    duration: Seconds = None
    
    def __post_init__(self):
        # round start and duration
        self.start = round(self.start, ndigits=8)
        self.duration = round(self.duration, ndigits=8)

    @property
    def end(self) -> Seconds:
        return round(self.start + self.duration, ndigits=8)

    def with_offset(self, offset: Seconds) -> "AlignmentDataSource":
        """Return an identical ``AlignmentDataSource``, but with the ``offset`` added to the ``start`` field."""
        return AlignmentDataSource(
            symbol=self.symbol, 
            start=round(self.start + offset, ndigits=8), 
            duration=self.duration
        )

    def perturb_speed(self, factor: float, sampling_rate: int) -> "AlignmentDataSource":
        """
        Return an ``AlignmentDataSource`` that has time boundaries matching the
        recording/cut perturbed with the same factor.
        """
        start_sample = compute_num_samples(self.start, sampling_rate)
        num_samples = compute_num_samples(self.duration, sampling_rate)
        new_start = round(perturb_num_samples(start_sample, factor) / sampling_rate, ndigits=8)
        new_duration = round(perturb_num_samples(num_samples, factor) / sampling_rate, ndigits=8)
        return AlignmentDataSource(
            symbol=self.symbol, 
            start=new_start, 
            duration=new_duration
        )

    def trim(self, end: Seconds, start: Seconds = 0) -> "AlignmentDataSource":
        """
        Return an identical ``AlignmentDataSource``, but ensure that ``self.start`` is not negative (in which case
        it's set to 0) and ``self.end`` does not exceed the ``end`` parameter. If a `start` is optionally
        provided, the data source is trimmed from the left (note that start should be relative to the cut times).

        This method is useful for ensuring that the data source does not exceed its bounds.
        """
        assert start >= 0
        start_exceeds_by = abs(min(0, self.start - start))
        end_exceeds_by = max(0, self.end - end)
        true_start = round(max(start, self.start), ndigits=8)
        true_duration = round(self.duration - end_exceeds_by - start_exceeds_by, ndigits=8)
        if true_start < 0.0 or true_duration < 0.0:
            return None
        return AlignmentDataSource(
            symbol=self.symbol,
            start=true_start,
            duration=true_duration,
        )

    def transform(self, transform_fn: Callable[[str], str]) -> "AlignmentDataSource":
        """
        Perform specified transformation on the alignment content.
        """
        return AlignmentDataSource(
            symbol=transform_fn(self.symbol), 
            start=self.start, 
            duration=self.duration
        )

