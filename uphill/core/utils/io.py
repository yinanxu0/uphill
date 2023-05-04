import os
import io
import gzip
import uuid
import tarfile
import hashlib
import yaml
import json
import sys
from pathlib import Path
from dataclasses import asdict
from typing import (
    Any,
    Callable,
    Dict,
    Optional,
    Iterable,
    Union
)


from uphill.errors import (
    FileNotExistError, 
    InvalidYAMLError, 
    InvalidJsonError, 
    InvalidJsonlError
)
from uphill import loggerx

from .constant import CHUNK_SIZE
from .typing import T, Pathlike, Manifest
from .module import is_module_available


Manifest = Any


class SmartOpen:
    """Wrapper class around smart_open.open method

    The smart_open.open attributes are cached as classed attributes - they play the role of singleton pattern.

    The SmartOpen.setup method is intended for initial setup.
    It imports the `open` method from the optional `smart_open` Python package,
    and sets the parameters which are shared between all calls of the `smart_open.open` method.

    If you do not call the setup method it is called automatically in SmartOpen.open with the provided parameters.

    The example demonstrates that instantiating S3 `session.client` once,
    instead using the defaults and leaving the smart_open creating it every time
    has dramatic performance benefits.

    Example::

        >>> import boto3
        >>> session = boto3.Session()
        >>> client = session.client('s3')
        >>> from uphill.core.utils import SmartOpen
        >>>
        >>> if not slow:
        >>>     # Reusing a single client speeds up the smart_open.open calls
        >>>     SmartOpen.setup(transport_params=dict(client=client))
        >>>
        >>> # Simulating SmartOpen usage easily.
        >>> for i in range(1000):
        >>>     SmartOpen.open(s3_url, 'rb') as f:
        >>>         source = f.read()
    """

    transport_params: Optional[Dict] = None
    import_err_msg = (
        "Please do 'pip install smart_open' - "
        "if you are using S3/GCP/Azure/other cloud-specific URIs, do "
        "'pip install smart_open[s3]' (or smart_open[gcp], etc.) instead."
    )
    smart_open: Optional[Callable] = None

    @classmethod
    def setup(cls, transport_params: Optional[dict] = None):
        try:
            from smart_open import open as sm_open
        except ImportError:
            raise ImportError(cls.import_err_msg)
        if (
            cls.transport_params is not None
            and cls.transport_params != transport_params
        ):
            loggerx.warning(
                f"SmartOpen.setup second call overwrites existing transport_params with new version"
                f"\t\n{cls.transport_params}\t\nvs\t\n{transport_params}"
            )
        cls.transport_params = transport_params
        cls.smart_open = sm_open

    @classmethod
    def open(cls, uri, mode="rb", transport_params=None, **kwargs):
        if cls.smart_open is None:
            cls.setup(transport_params=transport_params)
        transport_params = (
            transport_params if transport_params else cls.transport_params
        )
        return cls.smart_open(
            uri,
            mode=mode,
            transport_params=transport_params,
            **kwargs,
        )


def fastcopy(dataclass_obj: T, **kwargs) -> T:
    """
    Returns a new object with the same member values.
    Selected members can be overwritten with kwargs.
    It's supposed to work only with dataclasses.
    It's 10X faster than the other methods I've tried...

    Example:
        >>> ts1 = TimeSpan(start=5, end=10)
        >>> ts2 = fastcopy(ts1, end=12)
    """
    return type(dataclass_obj)(**{**dataclass_obj.__dict__, **kwargs})


def asdict_nonull(dclass) -> Dict[str, Any]:
    """
    Recursively convert a dataclass into a dict, removing all the fields with `None` value.
    Intended to use in place of dataclasses.asdict(), when the null values are not desired in the serialized document.
    """

    def non_null_dict_factory(collection):
        d = dict(collection)
        remove_keys = []
        for key, val in d.items():
            if val is None:
                remove_keys.append(key)
        for k in remove_keys:
            del d[k]
        return d

    return asdict(dclass, dict_factory=non_null_dict_factory)


def md5sum(fname: str):
    hash_md5 = hashlib.md5()
    loggerx.info(f"Compute md5 of {fname}")
    with open(fname, 'rb') as fp:
        for chunk in iter(lambda: fp.read(CHUNK_SIZE), b''):
            hash_md5.update(chunk)
    md5 = hash_md5.hexdigest()
    loggerx.debug(f"md5 of {fname} is {md5}")
    return md5


def touch_dir(dir_path: Pathlike):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def force_symlink(src, dst):
    if os.path.lexists(dst):
        os.remove(dst)
    os.symlink(src, dst)


def remove_extension(filepath):
    filename = os.path.basename(filepath)
    key = os.path.splitext(filename)[0]
    return key


def temp_file(dir_path: str, suffix=None):
    filename = str(uuid.uuid4())
    if suffix:
        filename += "." + suffix.lstrip(".")
    
    filepath = os.path.join(dir_path, filename)
    return filepath


def temp_dir(dir_path=None):
    if not dir_path:
        dir_path = "/tmp"
    dirname = str(uuid.uuid4())
    dirpath = os.path.join(dir_path, dirname)
    touch_dir(dirpath)
    return dirpath


def unpack(filepath: str, target_dir: str, keep_origin: bool = True):
    """Unpack the file to the target_dir."""
    loggerx.debug(f'Unpacking {filepath} to {target_dir}')
    with tarfile.open(filepath) as tar:
        tar.extractall(path=target_dir)
    if not keep_origin:
        os.remove(filepath)


def pack(dirpath: str, target_path: str, keep_origin: bool = True):
    """Pack the files to the tar file."""
    loggerx.debug(f'Packing {dirpath} to {target_path}')
    with tarfile.open(target_path, "w") as tar:
        tar.add(dirpath)
    if not keep_origin:
        os.remove(dirpath)


class AltGzipFile(gzip.GzipFile):
    """
    This is a workaround for Python's stdlib gzip module
    not implementing gzip decompression correctly...
    Command-line gzip is able to discard "trailing garbage" in gzipped files,
    but Python's gzip is not.

    Original source: https://gist.github.com/nczeczulin/474ffbf6a0ab67276a62
    """

    def read(self, size=-1):
        chunks = []
        try:
            if size < 0:
                while True:
                    chunk = self.read1()
                    if not chunk:
                        break
                    chunks.append(chunk)
            else:
                while size > 0:
                    chunk = self.read1(size)
                    if not chunk:
                        break
                    size -= len(chunk)
                    chunks.append(chunk)
        except OSError as e:
            if not chunks or not str(e).startswith("Not a gzipped file"):
                raise
            # loggerx.warn('decompression OK, trailing garbage ignored')

        return b"".join(chunks)


def gzip_open_robust(
    filename,
    mode="rb",
    compresslevel=9,  # compat with Py 3.6
    encoding=None,
    errors=None,
    newline=None,
):
    """Open a gzip-compressed file in binary or text mode.

    The filename argument can be an actual filename (a str or bytes object), or
    an existing file object to read from or write to.

    The mode argument can be "r", "rb", "w", "wb", "x", "xb", "a" or "ab" for
    binary mode, or "rt", "wt", "xt" or "at" for text mode. The default mode is
    "rb", and the default compresslevel is 9.

    For binary mode, this function is equivalent to the GzipFile constructor:
    GzipFile(filename, mode, compresslevel). In this case, the encoding, errors
    and newline arguments must not be provided.

    For text mode, a GzipFile object is created, and wrapped in an
    io.TextIOWrapper instance with the specified encoding, error handling
    behavior, and line ending(s).

    Note: This method is copied from Python's 3.7 stdlib, and patched to handle
    "trailing garbage" in gzip files. 
    """
    if "t" in mode:
        if "b" in mode:
            raise ValueError("Invalid mode: %r" % (mode,))
    else:
        if encoding is not None:
            raise ValueError("Argument 'encoding' not supported in binary mode")
        if errors is not None:
            raise ValueError("Argument 'errors' not supported in binary mode")
        if newline is not None:
            raise ValueError("Argument 'newline' not supported in binary mode")

    gz_mode = mode.replace("t", "")
    if isinstance(filename, (str, bytes, os.PathLike)):
        binary_file = AltGzipFile(filename, gz_mode, compresslevel)
    elif hasattr(filename, "read") or hasattr(filename, "write"):
        binary_file = AltGzipFile(None, gz_mode, compresslevel, filename)
    else:
        raise TypeError("filename must be a str or bytes object, or a file")

    if "t" in mode:
        return io.TextIOWrapper(binary_file, encoding, errors, newline)
    else:
        return binary_file


def open_best(path: Pathlike, mode: str = "r"):
    """
    Auto-determine the best way to open the input path or URI.
    Uses ``smart_open`` when available to handle URLs and URIs.
    Supports providing "-" as input to read from stdin or save to stdout.
    """
    class StdStreamWrapper:
        def __init__(self, stream):
            self.stream = stream

        def close(self):
            pass

        def __enter__(self):
            return self.stream

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

        def __getattr__(self, item: str):
            if item == "close":
                return self.close
            return getattr(self.stream, item)

    if path == "-":
        if mode == "r":
            return StdStreamWrapper(sys.stdin)
        elif mode == "w":
            return StdStreamWrapper(sys.stdout)
        else:
            raise ValueError(
                f"Cannot open stream for '-' with mode other 'r' or 'w' (got: '{mode}')"
            )

    if is_module_available("smart_open"):
        from smart_open import smart_open

        # This will work with JSONL anywhere that smart_open supports, e.g. cloud storage.
        open_fn = smart_open
    else:
        compressed = str(path).endswith(".gz")
        if compressed and "t" not in mode and "b" not in mode:
            # Opening as bytes not requested explicitly, use "t" to tell gzip to handle unicode.
            mode = mode + "t"
        open_fn = gzip_open_robust if compressed else open

    return open_fn(path, mode)


###########################
## IO functions for yaml ##
###########################
def load_yaml(path: Pathlike) -> dict:
    # check file exists
    if not os.path.exists(path):
        raise FileNotExistError(f"Not exists: {path}")
    
    loggerx.info(f"Loading yaml from {path}")
    docs = None
    with open_best(path) as fp:
        try:
            # When pyyaml is installed with C extensions, it can speed up the (de)serialization noticeably
            # yaml.FullLoader
            docs = yaml.load(stream=fp, Loader=yaml.CSafeLoader)
        except AttributeError:
            docs = yaml.load(stream=fp, Loader=yaml.SafeLoader)
    if not docs:
        raise InvalidYAMLError(f"Empty content: {path}")
    return docs

def save_yaml(data: Any, path: Pathlike) -> None:
    with open_best(path, "w") as fp:
        try:
            # When pyyaml is installed with C extensions, it can speed up the (de)serialization noticeably
            yaml.dump(data, stream=fp, sort_keys=False, Dumper=yaml.CSafeDumper)
        except AttributeError:
            yaml.dump(data, stream=fp, sort_keys=False, Dumper=yaml.SafeDumper)

###########################
## IO functions for json ##
###########################
def save_json(data: Any, path: Pathlike) -> None:
    """Save the data to a JSON file. Will use GZip to compress it if the path ends with a ``.gz`` extension."""
    with open_best(path, "w") as fp:
        fp.write(json.dumps(data, indent=2, ensure_ascii=False) + "\n")

def load_json(path: Pathlike) -> Union[dict, list]:
    """Load a JSON file. Also supports compressed JSON with a ``.gz`` extension."""
    if not os.path.exists(path):
        raise FileNotExistError(f"Not exists: {path}")
    
    content = None
    with open_best(path) as fp:
        try:
            content = json.load(fp)
        except json.decoder.JSONDecodeError:
            raise InvalidJsonError(f"Cannot decode content: {path}")
    return content

############################
## IO functions for jsonl ##
############################
def save_jsonl(data: Iterable[Dict[str, Any]], path: Pathlike) -> None:
    """Save the data to a JSON file. Will use GZip to compress it if the path ends with a ``.gz`` extension."""
    with open_best(path, "w") as fp:
        for item in data:
            fp.write(json.dumps(item, ensure_ascii=False) + "\n")

def load_jsonl(path: Pathlike, partition: str = "0/1", drop_last: bool = False) -> Iterable[Dict[str, Any]]:
    """Load a JSON file. Also supports compressed JSON with a ``.gz`` extension."""
    if not os.path.exists(path):
        raise FileNotExistError(f"Not exists: {path}")
    assert len(partition.split("/")) == 2, f"partition expects i/n format, gets {partition}"
    part_i, part_n = partition.split("/")
    part_i, part_n = int(part_i), int(part_n)
    assert part_n > 0, f"total partitions should be positive, gets n={part_n}"
    assert part_i < part_n, f"partition expects i in (0, {part_n-1}), gets i={part_i}"

    contents = []
    with open_best(path) as fp:
        if part_n == 1:
            contents = fp.readlines()
        else:
            if not drop_last:
                contents = [line for idx, line in enumerate(fp) if idx % part_n == part_i]
            else:
                last_line = None
                for idx, line in enumerate(fp):
                    if idx % part_n == part_i:
                        # update last_line
                        last_line = line
                    if idx % part_n == part_n-1 and last_line is not None:
                        # push last_line
                        contents.append(last_line)
    if len(contents) == 0:
        raise InvalidJsonlError(f"Empty content: {path}")
    return contents


def extension_contains(ext: str, path: Pathlike) -> bool:
    return any(ext == sfx for sfx in Path(path).suffixes)


def load_manifest(path: Pathlike, manifest_cls: Manifest = None, partition: str = "0/1", drop_last: bool = False) -> Optional[Manifest]:
    """Generic utility for reading an arbitrary manifest."""
    from uphill.document import Document, DataSource, Supervision
    from uphill.array import DocumentArray, SupervisionArray
    # Determine the serialization format and read the raw data.
    if extension_contains(".jsonl", path):
        raw_data = load_jsonl(path, partition=partition, drop_last=drop_last)
        if manifest_cls is None:
            # Note: for now, we need to load the whole JSONL rather than read it in
            # a streaming way, because we have no way to know which type of manifest
            # we should decode later; since we're consuming the underlying generator
            # each time we try, not materializing the list first could lead to data loss
            raw_data = list(raw_data)
    elif extension_contains(".json", path):
        raw_data = load_json(path)
    elif extension_contains(".yaml", path):
        raw_data = load_yaml(path)
    else:
        raise ValueError(f"Not a valid manifest (does the path exist?): {path}")
    data_set = None

    # If the user provided a "type hint", use it; otherwise we will try to guess it.
    if manifest_cls is not None:
        candidates = [manifest_cls]
    else:
        candidates = [Document, DataSource, Supervision, DocumentArray, SupervisionArray]
    for manifest_type in candidates:
        try:
            data_set = manifest_type.from_dict(raw_data)
            if len(data_set) == 0:
                raise RuntimeError()
            break
        except Exception:
            pass
    if data_set is None:
        raise ValueError(f"Unknown type of manifest: {path}")
    return data_set
