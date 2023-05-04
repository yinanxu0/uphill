import os
import pytest
from pathlib import Path

from uphill.document import DataSource, Document

test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

BLOB_LENGTH = {
    "A_16k_c1": 192012,
    "A_16k_c2": 383980
}
TENSOR_SHAPE = {
    "A_16k_c1": (1, 95984),
    "A_16k_c2": (2, 95984)
}
TYPE_DOC_MAP = {
    ".wav": "AudioDocument",
    ".txt": "TextDocument",
    ".mp4": "VideoDocument",
    ".jpg": "ImageDocument",
}

@pytest.mark.parametrize(
    'file_path', [
        os.path.join(test_dir, "toydata/wav/A_16k_c1.wav"), 
        # os.path.join(test_dir, "toydata/text/1.txt")
    ]
)
@pytest.mark.parametrize(
    'uid', ["A_16k_c1", None,]
)
def test_document_construction_from_uri(file_path, uid):
    doc = Document.from_uri(uri=file_path, id=uid)
    assert doc.__class__.__name__ == TYPE_DOC_MAP[Path(file_path).suffix]
    if uid is not None:
        assert doc.id == uid
    data = doc.load_audio()
    assert data.shape == TENSOR_SHAPE[Path(file_path).stem]


@pytest.mark.parametrize(
    'uri', [
        os.path.join(test_dir, "toydata/wav/A_16k_c1.wav"), 
    ]
)
def test_document_equality(uri):
    uri_path = Path(uri)
    uid = uri_path.stem
    doc_left = Document.from_uri(id=uid, uri=uri)
    doc_right = Document.from_uri(id=uid, uri=uri_path)
    
    doc_speed = doc_left.perturb_speed(1.1, affix_id=False)
    assert doc_left == doc_right
    assert doc_left != doc_speed
    
    doc_left_dict_info = doc_left.to_dict()
    doc_left_from_info = Document.from_dict(doc_left_dict_info)
    assert doc_left == doc_left_from_info
    
    doc_other = Document.from_uri(id=uid+"1", uri=uri_path)
    assert doc_left != doc_other


@pytest.mark.parametrize(
    'uri', [
        os.path.join(test_dir, "toydata/wav/A_16k_c1.wav"), 
    ]
)
@pytest.mark.parametrize('affix_id', [True, False])
def test_document_transform(uri, affix_id):
    uri_path = Path(uri)
    uid = uri_path.stem
    doc_src = Document.from_uri(id=uid, uri=uri_path)
    
    doc_speed = doc_src.perturb_speed(1.1, affix_id=affix_id)
    doc_volume = doc_speed.perturb_volume(2, affix_id=affix_id)
    doc_tempo = doc_volume.perturb_tempo(1.1, affix_id=affix_id)
    doc_resample = doc_tempo.resample(8000, affix_id=affix_id)
    
    if affix_id:
        assert doc_resample.id == uid + "_sp1.1_vp2_tp1.1_rs8000"
    else:
        assert doc_resample.id == uid
    
    doc_resample_dict = doc_resample.to_dict()
    doc_resample_from_dict = Document.from_dict(doc_resample_dict)
    assert doc_resample == doc_resample_from_dict
    
    audio = doc_resample_from_dict.load_audio()
    assert audio.shape == (1, 39663)


@pytest.mark.parametrize(
    'uri', [
        os.path.join(test_dir, "toydata/wav/A_16k_c1.wav"), 
    ]
)
def test_document_add_source(uri):
    uri_path = Path(uri)
    uid = uri_path.stem
    doc = Document.from_uri(id=uid, uri=uri_path)
    ds = DataSource.from_uri(uri_path)
    ds.num_channels = [0]
    
    tensor_shape = list(TENSOR_SHAPE[uid])

    # one source
    audio = doc.load_audio()
    assert audio.shape == tuple(tensor_shape)
    
    # two sources
    doc.add_source(ds)
    audio = doc.load_audio()
    tensor_shape[0] = 2
    assert audio.shape == tuple(tensor_shape)
    
    # three sources
    doc.add_source(ds)
    audio = doc.load_audio()
    tensor_shape[0] = 3
    assert audio.shape == tuple(tensor_shape)

