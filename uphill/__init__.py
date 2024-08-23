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

from uphill.version import (
    get_package_version,
    check_package_version
)

__version__ = get_package_version()
