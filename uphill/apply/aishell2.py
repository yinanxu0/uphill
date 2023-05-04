# """
# AISHELL2 (~1000 hours) if available(https://www.aishelltech.com/aishell_2).
# """
from collections import defaultdict
from pathlib import Path
import sys
from typing import Dict, Optional, Union

from uphill.core.utils import Pathlike
from uphill.core.text import Tokenizer
from uphill.document import Document, Supervision
from uphill.array import DocumentArray, SupervisionArray, TextDocumentArray
from uphill import loggerx


def prepare(
    corpus_dir: Pathlike,
    target_dir: Optional[Pathlike] = None,
    num_jobs: int = 1,
    compress: bool = False,
    remove_space: bool = True
) -> Dict[str, Dict[str, Union[DocumentArray, SupervisionArray]]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions
    :param corpus_dir: Pathlike, the path of the data dir.
    :param target_dir: Pathlike, the path where to write the manifests.
    :return: a Dict whose key is the dataset part, and the value is Dicts with the keys 'recordings' and 'supervisions'.
    """
    corpus_dir = Path(corpus_dir)
    dataset_parts = ["train", "dev", "test"]
    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"
    
    for part in dataset_parts:
        part_alias = part if part == "train" else f"iOS/{part}"
        sub_dir = corpus_dir / part_alias
        error_message = f"contents in {corpus_dir} should be like following: "
        error_message += f"\n{corpus_dir}/"
        error_message += f"\n├── train/"
        error_message += f"\n│   ├── trans.txt"
        error_message += f"\n│   └── wav/"
        error_message += f"\n├── iOS/"
        error_message += f"\n│   ├── dev/"
        error_message += f"\n│   │   ├── trans.txt"
        error_message += f"\n│   │   └── wav/"
        error_message += f"\n│   └── test/"
        error_message += f"\n│       ├── trans.txt"
        error_message += f"\n│       └── wav/"
        error_message += f"\n├── Mic" 
        error_message += f"\n├── Android" 
        error_message += f"\n└── ... (other files)"
        if not sub_dir.is_dir() or not sub_dir.exists():
            loggerx.warning(error_message)
            sys.exit()
    
    if target_dir is not None:
        target_dir = Path(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

    manifests = defaultdict(dict)
    for part in dataset_parts:
        loggerx.info(f"Processing aishell2 subset: {part}")
        part_alias = f"iOS/{part}" if part != "train" else part
        transcript_path = corpus_dir / f"{part_alias}/trans.txt"                               
        wav_path = corpus_dir / f"{part_alias}/wav"

        da_text = TextDocumentArray()
        with open(transcript_path, "r", encoding="utf-8") as f:
            joiner = "" if remove_space else " "
            for line in f.readlines():
                idx_transcript = line.strip().split()
                idx_transcript[0] = str(Path(idx_transcript[0]).stem)
                doc_text = Document.from_text(
                    id = idx_transcript[0],
                    text=joiner.join(idx_transcript[1:])
                )
                da_text.append(doc_text)
        
        tokenizer = Tokenizer()
        vocab = da_text.get_vocab(tokenizer, num_jobs=num_jobs)
        vocab.insert(0, '<blank>')
        vocab.insert(1, '<unk>')
        vocab.append('<sos/eos>')
        vocab.to_file(target_dir / "words.txt")
        
        da_wav = DocumentArray.from_dir(wav_path, pattern="*.wav", num_jobs=num_jobs)
        sa = SupervisionArray()
            
        for audio_path in wav_path.rglob("*.wav"):
            idx = audio_path.stem
            speaker = audio_path.parts[-2]
            if idx not in da_text:
                loggerx.warning(f"No transcript: {idx}")
                continue
            doc_text = da_text[idx]
            if not audio_path.is_file():
                loggerx.warning(f"No such file: {audio_path}")
                continue
            doc_wav = da_wav[idx]
            # add custom attributes to wav document
            doc_wav.speaker = speaker
            doc_wav.language = "Chinese"
            sa.append(
                Supervision.from_document(
                    source=doc_wav,
                    target=doc_text,
                    id=idx
                )
            )
            if part == "train":
                for speed in [0.9, 1.1]:
                    doc_wav_sp_aug = doc_wav.perturb_speed(speed)
                    da_wav.append(doc_wav_sp_aug)
                    sa.append(
                        Supervision.from_document(
                            source=doc_wav_sp_aug,
                            target=doc_text,
                            id=f"{idx}_sp{speed}"
                        )
                    )
        
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

