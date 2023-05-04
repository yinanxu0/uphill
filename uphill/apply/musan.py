import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional


from uphill.core.utils import (
    download_url,
    unpack,
    Pathlike,
)
from uphill.array import DocumentArray
from uphill import loggerx


def download(
    target_dir: Pathlike = ".",
    force_download: bool = False,
    base_url: str = "http://www.openslr.org/resources/17",
) -> Path:
    """
    Downdload and untar the dataset
    :param target_dir: Pathlike, the path of the dir to storage the dataset.
    :param force_download: Bool, if True, download the tars no matter if the tars exist.
    :param base_url: str, the url of the OpenSLR resources.
    :return: the path to downloaded and extracted directory with data.
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    corpus_dir = target_dir / "musan"
    tar_dir = target_dir / "musan_tar"
    
    dataset_tar_name = "musan.tar.gz"
    dataset_info = {
        dataset_tar_name: "0c472d4fc0c5141eca47ad1ffeb2a7df"
    }
    for tar_name, md5 in dataset_info.items():
        extracted_dir = corpus_dir.parents[0]
        completed_detector = corpus_dir / ".completed"
        if completed_detector.is_file():
            loggerx.info(f"Skipping download of {tar_name} because completed detector exists.")
            continue
        tar_path = download_url(
            url=f"{base_url}/{tar_name}", 
            md5=md5, 
            target_dir=tar_dir, 
            force_download=force_download
        )
        shutil.rmtree(corpus_dir, ignore_errors=True)
        unpack(tar_path, extracted_dir)
        completed_detector.touch()
    return corpus_dir



def prepare(
    corpus_dir: Pathlike, 
    target_dir: Optional[Pathlike] = None, 
    num_jobs: int = 1,
    compress: bool = False,
) -> Dict[str, Dict[str, DocumentArray]]:
    """
    Returns the manifests which consist of the Utterances and Supervisions
    :param corpus_dir: Pathlike, the path of the data dir.
    :param target_dir: Pathlike, the path where to write the manifests.
    :return: a Dict whose key is the dataset part, and the value is Dicts with the keys 'wav_documents' and 'supervisions'.
    """
    corpus_dir = Path(corpus_dir)
    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"
    if target_dir is not None:
        target_dir = Path(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
    
    manifests = defaultdict(dict)
    da_wav = DocumentArray.from_dir(corpus_dir, pattern="**/**/*.wav", num_jobs=num_jobs)
    da_noise_wav = da_wav.filter(lambda doc: not doc.id.startswith("speech"))
    if target_dir is not None:
        target_extension = ""
        if compress:
            target_extension = ".gz"
        da_wav.to_file(target_dir / f"wav_documents_musan_full.jsonl{target_extension}")
        da_noise_wav.to_file(target_dir / f"wav_documents_musan_noise.jsonl{target_extension}")
        manifests = {
            "wav_documents": da_wav, 
            "noise_wav_documents": da_noise_wav, 
        }
    return manifests

