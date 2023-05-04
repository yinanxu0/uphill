import numpy as np
from pathlib import Path
from typing import (
    Any,
    Callable,
    List,
    Sequence,
    TypeVar,
    Union,
    ForwardRef
)

import scipy.sparse
# import tensorflow
import torch
import numpy as np


##############
# type alias #
##############
Pathlike = Union[Path, str]
Channels = Union[int, List[int]]
Seconds = float
Decibels = float
FileObject = Any
Manifest = Any
T = TypeVar("T")

Image = TypeVar(
    'Image',
    str,
    ForwardRef('np.ndarray'),
    ForwardRef('PILImage'),
)
Text = TypeVar('Text', bound=str)
Audio = TypeVar('Audio', str, ForwardRef('np.ndarray'))
Video = TypeVar('Video', str, ForwardRef('np.ndarray'))
Mesh = TypeVar('Mesh', str, ForwardRef('np.ndarray'))
Tabular = TypeVar('Tabular', bound=str)
Blob = TypeVar('Blob', str, bytes)
JSON = TypeVar('JSON', str, dict)
JSONL = TypeVar('JSONL', str, List[dict])



ArrayType = TypeVar(
    'ArrayType',
    np.ndarray,
    scipy.sparse.spmatrix,
    # tensorflow.SparseTensor,
    # tensorflow.Tensor,
    torch.Tensor,
    Sequence[float],
)



######################
# callable functions #
######################
AugmentFn = Callable[[np.ndarray, int], np.ndarray]


####################
# useful functions #
####################
import sys
if sys.version_info >= (3, 8, 0):
    from typing import get_args
else:
    def get_args(data: Any):
        return getattr(data, "__args__", None) 
