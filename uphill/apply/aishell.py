import shutil
from functools import partial
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Union

from uphill.core.utils import (
    download_url,
    unpack,
    Pathlike,
    parallel_run
)
from uphill.core.text import Tokenizer
from uphill.document import Document, Supervision
from uphill.array import DocumentArray, SupervisionArray, TextDocumentArray
from uphill import loggerx


def download(
    target_dir: Pathlike = ".",
    force_download: bool = False,
    base_url: str = "http://www.openslr.org/resources/33",
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
    corpus_dir = target_dir / "aishell"
    tar_dir = target_dir / "aishell_tar"
    
    dataset_tar_name = "data_aishell.tgz"
    resources_tar_name = "resource_aishell.tgz"
    dataset_info = {
        dataset_tar_name: "2f494334227864a8a8fec932999db9d8",
        resources_tar_name: "957d480a0fcac85fc18e550756f624e5"
    }
    for tar_name, md5 in dataset_info.items():
        extracted_dir = corpus_dir / Path(tar_name).stem
        completed_detector = extracted_dir / ".completed"
        if completed_detector.is_file():
            loggerx.info(f"Skipping download of {tar_name} because completed detector exists.")
            continue
        tar_path = download_url(
            url=f"{base_url}/{tar_name}", 
            md5=md5, 
            target_dir=tar_dir, 
            force_download=force_download
        )
        shutil.rmtree(extracted_dir, ignore_errors=True)
        unpack(tar_path, corpus_dir)
        if tar_name == dataset_tar_name:
            wav_dir = extracted_dir / "wav"
            wav_tar_files = []
            for sub_file_or_folder in wav_dir.iterdir():
                if not sub_file_or_folder.is_dir():
                    wav_tar_files.append(sub_file_or_folder)
            for name in ["train", "dev", "test"]:
                sub_dir = wav_dir / name
                sub_dir.mkdir(parents=True, exist_ok=True)
            unpack_fn = partial(unpack, target_dir=wav_dir)
            parallel_run(unpack_fn, wav_tar_files, num_jobs=10)
        completed_detector.touch()
    return corpus_dir


def prepare(
    corpus_dir: Pathlike, 
    target_dir: Optional[Pathlike] = None, 
    num_jobs: int = 1,
    compress: bool = False,
    remove_space: bool = True
) -> Dict[str, Dict[str, Union[DocumentArray, SupervisionArray]]]:
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
    transcript_path = corpus_dir / "data_aishell/transcript/aishell_transcript_v0.8.txt"
    da_text_all = TextDocumentArray()
    with open(transcript_path, "r", encoding="utf-8") as f:
        joiner = "" if remove_space else " "
        for line in f.readlines():
            idx_transcript = line.strip().split()
            doc_text = Document.from_text(
                id = idx_transcript[0],
                text=joiner.join(idx_transcript[1:])
            )
            da_text_all.append(doc_text)

    tokenizer = Tokenizer()
    vocab = da_text_all.get_vocab(tokenizer)
    vocab.insert(0, '<blank>')
    vocab.insert(1, '<unk>')
    vocab.append('<sos/eos>')
    vocab.to_file(target_dir / "words.txt")
            
    manifests = defaultdict(dict)
    dataset_parts = ["train", "dev", "test"]
    for part in dataset_parts:
        loggerx.info(f"Processing aishell subset: {part}")
        wav_path = corpus_dir / "data_aishell/wav" / part
        da_wav = DocumentArray.from_dir(wav_path, pattern="**/*.wav", num_jobs=num_jobs)
        da_text = TextDocumentArray()
        sa = SupervisionArray()
        
        for audio_path in wav_path.rglob("**/*.wav"):
            idx = audio_path.stem
            speaker = audio_path.parts[-2]
            if idx not in da_text_all:
                loggerx.warning(f"No transcript: {idx}")
                continue
            doc_text = da_text_all[idx]
            da_text.append(doc_text)
            if not audio_path.is_file():
                loggerx.warning(f"No such file: {audio_path}")
                continue
            doc_wav = da_wav[idx]
            # add custom attributes to wav document
            doc_wav.speaker = speaker
            doc_wav.language = "Chinese"
            supervision = Supervision.from_document(
                source=doc_wav,
                target=doc_text,
                id=idx
            )
            sa.append(supervision)
            if part == "train":
                for speed in [0.9, 1.1]:
                    doc_wav_sp_aug = doc_wav.perturb_speed(speed)
                    da_wav.append(doc_wav_sp_aug)
                    sa.append(Supervision.from_document(
                        source=doc_wav_sp_aug,
                        target=doc_text,
                        id=idx + f"_sp{speed}"
                    ))
        
        if target_dir is not None:
            target_extension = ""
            if compress:
                target_extension = ".gz"
            sa.to_file(target_dir / f"supervisions_{part}.jsonl{target_extension}")
            da_wav.to_file(target_dir / f"wav_documents_{part}.jsonl{target_extension}")
            da_text.to_file(target_dir / f"text_documents_{part}.jsonl{target_extension}")
        manifests[part] = {
            "wav_documents": da_wav, 
            "text_documents": da_text, 
            "supervisions": sa,
        }
    return manifests
