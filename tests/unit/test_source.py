import os
import pytest
from pathlib import Path
import torchaudio

from uphill.document import DataSource


test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
BLOB_LENGTH = {
    "A_16k_c1": 192012,
    "A_16k_c2": 383980
}
TENSOR_SHAPE = {
    "A_16k_c1": (1, 95984),
    "A_16k_c2": (2, 95984)
}

@pytest.mark.parametrize(
    'uri', [
        os.path.join(test_dir, "toydata/wav/A_16k_c1.wav"), 
        os.path.join(test_dir, "toydata/wav/A_16k_c2.wav"), 
        os.path.join(test_dir, "toydata/wav/empty.wav")
    ]
)
def test_construct_datasource_from_uri(uri):
    uri_path = Path(uri)
    uid = uri_path.stem
    ds = DataSource.from_uri(id=uid, uri=uri_path)
    assert ds.id == uid
    if uid == "empty":
        try:
            ds.load_blob()
        except FileNotFoundError as error_message:
            print(error_message)
        return 
    ds.load_blob()
    assert len(ds.blob) == BLOB_LENGTH[uid]
    ds.load_tensor()
    assert ds.tensor.shape == TENSOR_SHAPE[uid]


@pytest.mark.parametrize(
    'uri', [
        os.path.join(test_dir, "toydata/wav/A_16k_c1.wav"), 
        os.path.join(test_dir, "toydata/wav/A_16k_c2.wav"), 
    ]
)
def test_construct_datasource_from_blob(uri):
    blob = None
    with open(uri, 'rb') as fp:
        blob = fp.read()
        
    uri_path = Path(uri)
    uid = uri_path.stem
    ds = DataSource.from_blob(id=uid, blob=blob, mime_type="audio")
    assert ds.id == uid
    ds.load_blob()
    assert len(ds.blob) == BLOB_LENGTH[uid]
    ds.load_tensor()
    assert ds.tensor.shape == TENSOR_SHAPE[uid]


@pytest.mark.parametrize(
    'uri', [
        os.path.join(test_dir, "toydata/wav/A_16k_c1.wav"), 
        os.path.join(test_dir, "toydata/wav/A_16k_c2.wav"), 
    ]
)
def test_construct_datasource_from_tensor(uri):
    tensor, sampling_rate = torchaudio.load(uri)
    uri_path = Path(uri)
    uid = uri_path.stem
    ds = DataSource.from_tensor(id=uid, tensor=tensor, sampling_rate=sampling_rate, mime_type="audio")
    assert ds.id == uid
    ds.load_blob()
    assert len(ds.blob) == BLOB_LENGTH[uid]
    ds.load_tensor()
    assert ds.tensor.shape == TENSOR_SHAPE[uid]


@pytest.mark.parametrize(
    'uri', [
        os.path.join(test_dir, "toydata/wav/A_16k_c1.wav"), 
    ]
)
def test_datasource_equality(uri):
    uri_path = Path(uri)
    uid = uri_path.stem
    ds_left = DataSource.from_uri(id=uid, uri=uri)
    ds_right = DataSource.from_uri(id=uid, uri=uri_path)
    
    ds_left.load_blob()
    ds_left.load_tensor()
    assert ds_left == ds_right
    
    ds_left_dict_info = ds_left.to_dict()
    
    ds_left_from_info = DataSource.from_dict(ds_left_dict_info)
    assert ds_left == ds_left_from_info
    
    ds_other = DataSource(id=uid+"1", uri=uri)
    assert ds_left != ds_other





