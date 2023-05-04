from uphill.logx import loggerx

from . import apply
from . import array
from . import bin
from . import core
from . import document


from uphill.document import (
    # Document related
    Document,
    AudioDocument,
    TextDocument,
    ImageDocument,
    VideoDocument,
    AlignmentDocument,
    # DataSource related
    DataSource,
    AudioDataSource,
    TextDataSource,
    ImageDataSource,
    VideoDataSource,
    AlignmentDataSource,
    # Supervision related
    Supervision,
)

from uphill.array import (
    # DocumentArray related
    DocumentArray,
    AudioDocumentArray,
    TextDocumentArray,
    ImageDocumentArray,
    VideoDocumentArray, 
    AlignmentDocumentArray,
    # SupervisionArray related
    SupervisionArray,
)


def get_package_version():
    import os
    version_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "VERSION")
    version = '0.0.0'
    for content in open(version_file, 'r', encoding='utf8').readlines():
        content = content.strip()
        if len(content) > 0:
            version = content
            break
    return version

__version__ = get_package_version()
