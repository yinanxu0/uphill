import os
import pytest
from pathlib import Path

from uphill.document import DataSource, Document
from uphill.array import DocumentArray

test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

@pytest.mark.parametrize(
    'dir_path_and_pattern', [
        (os.path.join(test_dir, "toydata/wav"), "*.wav"),
        (os.path.join(test_dir, "toydata/text"), "*.txt"),
    ]
)
@pytest.mark.parametrize(
    'use_doc_id_fn', [True, False]
)
def test_documentarray_construction_from_dir(dir_path_and_pattern, use_doc_id_fn):
    dir_path, pattern = dir_path_and_pattern
    def id_fn(path):
        return "test_" + Path(path).stem
    doc_id_fn = id_fn if use_doc_id_fn else None
    da = DocumentArray.from_dir(dir_path, pattern=pattern, document_id=doc_id_fn)
    for doc in da:
        if use_doc_id_fn:
            assert doc.id == "test_" + Path(doc.sources[0].uri).stem
        else:
            assert doc.id == Path(doc.sources[0].uri).stem


@pytest.mark.parametrize(
    'dir_path_and_pattern', [
        (os.path.join(test_dir, "toydata/wav"), "*.wav"),
        (os.path.join(test_dir, "toydata/text"), "*.txt"),
    ]
)
def test_documentarray_split_and_subset(dir_path_and_pattern):
    dir_path, pattern = dir_path_and_pattern
    da = DocumentArray.from_dir(dir_path, pattern=pattern)
    
    ## split function
    da_splits = da.split(len(da))
    assert len(da_splits) == len(da)
    for da_split in da_splits:
        assert len(da_split) == 1
    
    ## subset function
    da_first = da.subset(first=max(len(da)//2, 1))
    da_last = da.subset(last=max(len(da)//2, 1))
    assert len(da_first) == max(len(da)//2, 1)
    assert len(da_last) == max(len(da)//2, 1)


@pytest.mark.parametrize(
    'dir_path_and_pattern', [
        (os.path.join(test_dir, "toydata/wav"), "*.wav"),
    ]
)
def test_documentarray_access_document(dir_path_and_pattern):
    dir_path, pattern = dir_path_and_pattern
    da = DocumentArray.from_dir(dir_path, pattern=pattern)
    
    assert da.num_channels("A_16k_c1") == [0]
    assert da.sampling_rate("A_16k_c1") == 16000
    assert da.num_samples("A_16k_c1") == 95984
    assert da.duration("A_16k_c1") == 5.999


@pytest.mark.parametrize(
    'dir_path_and_pattern', [
        (os.path.join(test_dir, "toydata/wav"), "*.wav"),
    ]
)
@pytest.mark.parametrize('affix_id', [True, False])
def test_documentarray_transform(dir_path_and_pattern, affix_id):
    uid = "A_16k_c1"
    dir_path, pattern = dir_path_and_pattern
    
    da = DocumentArray.from_dir(dir_path, pattern=pattern)
    da_speed = da.perturb_speed(1.1, affix_id=affix_id)
    da_volume = da_speed.perturb_volume(2, affix_id=affix_id)
    da_tempo = da_volume.perturb_tempo(1.1, affix_id=affix_id)
    da_resample = da_tempo.resample(8000, affix_id=affix_id)

    if affix_id:
        doc_id = uid + "_sp1.1_vp2_tp1.1_rs8000"
    else:
        doc_id = uid
    
    assert doc_id in da_resample
    doc = da_resample[doc_id]
    audio_from_doc = doc.load_audio()
    audio_from_da = da_resample.load_audio(doc_id)
    assert audio_from_doc.shape == audio_from_da.shape == (1, 39663)

