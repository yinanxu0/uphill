import os
import urllib.parse
import urllib.request
from contextlib import nullcontext
from typing import Optional, Any
from typeguard import typechecked
from base64 import encodebytes as encode64


@typechecked
def _uri_to_blob(uri: str) -> bytes:
    """Convert uri to blob
    Internally it reads uri into blob.

    :param uri: the uri of Document
    :return: blob bytes.
    """
    if urllib.parse.urlparse(uri).scheme in {'http', 'https', 'data'}:
        req = urllib.request.Request(uri, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as fp:
            return fp.read()
    elif os.path.exists(uri):
        with open(uri, 'rb') as fp:
            return fp.read()
    else:
        raise FileNotFoundError(f'`{uri}` is not a valid URL local path')

def _get_file_context(file):
    if hasattr(file, 'write'):
        file_ctx = nullcontext(file)
    else:
        file_ctx = open(file, 'wb')

    return file_ctx


def _to_datauri(
    mimetype, data, charset: str = 'utf-8', base64: bool = False, binary: bool = True
) -> str:
    """
    Convert data to data URI.

    :param mimetype: MIME types (e.g. 'text/plain','image/png' etc.)
    :param data: Data representations.
    :param charset: Charset may be any character set registered with IANA
    :param base64: Used to encode arbitrary octet sequences into a form that satisfies the rules of 7bit. Designed to be efficient for non-text 8 bit and binary data. Sometimes used for text data that frequently uses non-US-ASCII characters.
    :param binary: True if from binary data False for other data (e.g. text)
    :return: URI data
    """
    parts = ['data:', mimetype]
    if charset is not None:
        parts.extend([';charset=', charset])
    if base64:
        parts.append(';base64')
        

        if binary:
            encoded_data = encode64(data).decode(charset).replace('\n', '').strip()
        else:
            encoded_data = encode64(data).strip()
    else:
        if binary:
            encoded_data = urllib.parse.quote_from_bytes(data)
        else:
            encoded_data = urllib.parse.quote(data)
    parts.extend([',', encoded_data])
    return ''.join(parts)


def _is_uri(value: str) -> bool:
    scheme = urllib.parse.urlparse(value).scheme
    return (
        (scheme in {'http', 'https'})
        or (scheme in {'data'})
        or os.path.exists(value)
        or os.access(os.path.dirname(value), os.W_OK)
    )

def _is_datauri(value: str) -> bool:
    scheme = urllib.parse.urlparse(value).scheme
    return scheme in {'data'}


def ifnone(item: Optional[Any], alt_item: Any) -> Any:
    """Return ``alt_item`` if ``item is None``, otherwise ``item``."""
    return alt_item if item is None else item