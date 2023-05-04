import os
import pytest
from pathlib import Path

import torch
import torchaudio
import numpy as np

from uphill.document import DataSource, Document, Supervision
from uphill.array import DocumentArray

test_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(test_dir, "toydata")
wav_path = os.path.join(data_dir, "wav/A_16k_c1.wav")
pattern = "*.wav"
def document_id_fn(path):
    return Path(path).stem

da = DocumentArray.from_dir(data_dir, pattern=pattern, document_id=None)

def test_fn():
    return os.urandom(4).hex()

for a,b in zip(da, da):
    instance = Supervision.from_document(source=a, target=b, id=test_fn)
    instance_dict = instance.to_dict()
    print(instance_dict)
    instance_from_dict = Supervision.from_dict(instance_dict)
    assert instance == instance_from_dict
    break

