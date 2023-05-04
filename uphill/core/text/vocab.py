import os
import re
from copy import deepcopy
from typing import Dict, Iterable, List, Optional, Union
import sentencepiece as spm
from typeguard import typechecked


from uphill.core.text import (
    guess_language,
    LANGUAGE, 
    BPE_DELIMITER
)
from uphill.core.utils import Pathlike, T
from uphill.errors import (
    FileNotExistError,
    BadFormatError, 
    BadInputError
)
from uphill import loggerx


class Tokenizer:
    @typechecked
    def __init__(
        self, 
        bpe_model: Optional[Pathlike] = None, 
        special_tokens: Optional[List[str]] = None,
        pad_token: str = "<pad>",
        bos_token: str = "<bos>",
        eos_token: str = "<eos>",
        unk_token: str = "<unk>",
    ) -> None:
        # default
        self._special_token_pattern = re.compile(r"(\[[^\[\]]+\]|<[^<>]+>|{[^{}]+})")
        self._bpe_delimiter = BPE_DELIMITER

        if bpe_model is not None:
            self._bpe_model = spm.SentencePieceProcessor()
            self._bpe_model.Load(bpe_model)
        else:
            self._bpe_model = None
    
        self._pad_token = pad_token if self._validate_special_token(pad_token) else None
        self._bos_token = bos_token if self._validate_special_token(bos_token) else None
        self._eos_token = eos_token if self._validate_special_token(eos_token) else None
        self._unk_token = unk_token if self._validate_special_token(unk_token) else None
        
        self._special_tokens = None if special_tokens is None else []
        if special_tokens is not None:
            for special_token in special_tokens:
                if not self._validate_special_token(special_token):
                    raise BadFormatError(
                        "Non-linguistic symbols should be "
                        "formatted in {xxx}/<xxx>/[xxx], consider "
                        f"modify {special_token} to meet the requirment.")
                self._special_tokens.append(special_token)
        
        self._predefine_special_tokens = []
        for token in [self._pad_token, self._bos_token, self._eos_token, self._unk_token]:
            if token is not None:
                self._predefine_special_tokens.append(token)

    
    ###################
    # property getter #
    ###################
    @property
    def special_tokens(self):
        return self._special_tokens
    
    @property
    def pad_token(self):
        return self._pad_token

    @property
    def bos_token(self):
        return self._bos_token
    
    @property
    def eos_token(self):
        return self._eos_token
    
    @property
    def unk_token(self):
        return self._unk_token
    
    ##############
    # operations #
    ##############
    def detokenize(self, tokens: List[str]) -> str:
        '''
        tokens to text
        '''
        fake_text = " ".join(tokens)
        language = guess_language(fake_text)
        if self._bpe_model is not None and language == LANGUAGE.English:
            # use bpe mode to decode
            return "".join(self._bpe_model.DecodePieces(tokens))
        
        text = " ".join(tokens)
        if guess_language(text) is LANGUAGE.Chinese:
            text = text.replace(" ", "")
        if self._bpe_delimiter in text:
            text = text.replace(self._bpe_delimiter, " ")
        return text
    
    def tokenize(self, text) -> List[str]:
        '''
        text to tokens
        '''
        text = text.strip().lower()
        text_language = guess_language(text)
        tokens = []
        
        # split text by special tokens
        if self._special_tokens is not None and len(self._special_tokens) > 0:
            sub_parts = [
                token 
                for token in self._special_token_pattern.split(text) 
                if len(token.strip()) > 0
            ]
        else:
            sub_parts = [text]
        
        # split chinese and english
        non_english_pattern = re.compile(r'([\u4e00-\u9fff])')
        sub_parts = [
            token 
            for sub_part in sub_parts
            for token in non_english_pattern.split(sub_part) 
            if len(token.strip()) > 0
        ]
        
        # combine continus same language parts
        sub_parts_bak = sub_parts
        sub_parts = []
        i = 0
        j = 0
        prev_language = None
        while j < len(sub_parts_bak):
            if prev_language is None:
                prev_language = guess_language(sub_parts_bak[i])
            
            curr_language = guess_language(sub_parts_bak[j])
            if curr_language != prev_language:
                sub_parts.append("".join(sub_parts_bak[i:j]))
                prev_language = curr_language
                i = j
            if j == len(sub_parts_bak)-1:
                sub_parts.append("".join(sub_parts_bak[i:]))
            j += 1
        
        # add bpe delimiter between sub part
        sub_parts_bak = sub_parts
        sub_parts = []
        for idx, sub_part in enumerate(sub_parts_bak):
            sub_parts.append(sub_part)
            if idx < len(sub_parts_bak) - 1:
                sub_parts.append(self._bpe_delimiter)
        
        for sub_part in sub_parts:
            sub_part = sub_part.strip()
            if sub_part == self._bpe_delimiter:
                tokens.append(sub_part)
                continue
            
            if self._special_tokens is not None and sub_part in self._special_tokens:
                tokens.append(sub_part)
            else:
                if guess_language(sub_part) == LANGUAGE.English:
                    if self._bpe_model is not None:
                        # use bpe mode to decode, language should be English
                        tokens.extend(self._bpe_model.EncodeAsPieces(sub_part))
                    else:
                        sub_tokens = []
                        for idx, sub_token in enumerate(sub_part.split()):
                            sub_tokens.append(sub_token)
                            sub_tokens.append(self._bpe_delimiter)
                        tokens.extend(sub_tokens[:-1])
                else:
                    if text_language == LANGUAGE.English and self._bpe_model is not None:
                        tokens.append(self._bpe_delimiter)
                    for token in sub_part:
                        if token == " ":
                            token = self._bpe_delimiter
                        tokens.append(token)
        return tokens

    
    ######################
    # internal functions #
    ######################
    def _validate_special_token(self, token):
        special_token_pattern = self._special_token_pattern
        if special_token_pattern.fullmatch(token) is None:
            return False
        return True


class Vocabulary(object):
    def __init__(
        self, 
        vocab: Optional[Dict[str, int]] = None,
        vocab_path: Optional[str] = None, 
    ) -> None:
        self._special_token_pattern = re.compile(r"(\[[^\[\]]+\]|<[^<>]+>|{[^{}]+})")
        if vocab is not None and vocab_path is not None:
            raise BadInputError(
                f"Inputing both vocab and vocab_path is not valid."
            )
        elif vocab_path is not None:
            vocab = self._load_vocab(vocab_path=vocab_path)
        elif vocab is not None:
            vocab = vocab
        else:
            vocab = {}
        self._vocab = self._validate_vocab(vocab)
    
    def _load_vocab(self, vocab_path):
        if vocab_path is None:
            return {}
        if not os.path.exists(vocab_path):
            raise FileNotExistError(f"{vocab_path} not exist. Please check!")
        vocab = {}
        with open(vocab_path) as fp:
            for idx, line in enumerate(fp.readlines()):
                elems = line.strip().split()
                if len(elems) != 2:
                    loggerx.warning(f"Vocabulary lineno {idx}: {line.strip()} not valid. Please check!")
                vocab[elems[0]] = int(elems[1])
        return vocab
    
    def _validate_vocab(self, vocab):
        valid_vocab = {}
        for token, idx in vocab.items():
            if self._special_token_pattern.fullmatch(token) is not None:
                # convert all special token to <token> format for convience
                token = token.replace(token[0], '<')
                token = token.replace(token[-1], '>')
            valid_vocab[token] = idx
        return valid_vocab
    
    @staticmethod
    def from_file(vocab_path: Optional[str] = None):
        return Vocabulary(vocab_path=vocab_path)
    
    @staticmethod
    def from_dict(vocab: Dict[str, int]):
        return Vocabulary(vocab=vocab)
    
    ###################
    # property getter #
    ###################
    @property
    def vocab(self):
        return self._vocab 
    
    @property
    def reverse_vocab(self):
        vocab = {}
        for token, idx in self._vocab.items():
            vocab[idx] = token
        return vocab

    @property
    def special_tokens(self):
        tokens = {}
        for token, idx in self._vocab.items():
            if self._special_token_pattern.fullmatch(token) is not None:
                tokens[token] = idx
        return tokens
    
    ##############
    # operations #
    ##############
    def append(self, token: str):
        if token in self._vocab:
            # loggerx.warning(f"{token} exists in vocab. Cannot apped to vocab.")
            return
        self._vocab[token] = len(self._vocab)
    
    def extend(self, tokens: Union[List[str], str]):
        for token in tokens:
            if token in self._vocab:
                # loggerx.warning(f"{token} exists in vocab. Cannot apped to vocab.")
                continue
            self._vocab[token] = len(self._vocab)
    
    def insert(self, idx: int, token: str):
        vocab_updated = {}
        for token_, idx_ in self._vocab.items():
            if idx_ >= idx:
                vocab_updated[token_] = idx_ + 1
            else:
                vocab_updated[token_] = idx_
        vocab_updated[token] = idx
        self._vocab = vocab_updated
    
    def remove(self, token: str):
        idx = self.token2id(token)
        if idx < 0:
            return
        vocab_updated = {}
        for token_, idx_ in self._vocab.items():
            if token_ == token:
                continue
            if idx_ >= idx:
                vocab_updated[token_] = idx_ - 1
            else:
                vocab_updated[token_] = idx_
        self._vocab = vocab_updated
    
    def token2id(self, token: str) -> int:
        if token not in self._vocab:
            if '<unk>' is not None:
                return self._vocab['<unk>']
            else:
                loggerx.warning(f"{token} and <unk> both not exists in vocab.")
                return -1
        return self._vocab[token]
    
    def id2token(self, idx: int) -> str:
        if idx < 0 or idx > len(self._vocab):
            loggerx.warning(f"{idx} not exists in vocab.")
            return ""
        for token, idx_ in self._vocab.items():
            if idx == idx_:
                return token
        loggerx.warning(f"{idx} not exists in vocab.")
        return ""
    
    def to_file(self, filepath: Pathlike):
        with open(filepath, "w") as fp:
            for idx in range(len(self._vocab)):
                token = self.id2token(idx)
                line = f"{token} {idx}\n"
                fp.write(line)


    ######################
    # internal functions #
    ######################
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(len={len(self._vocab)}, special_tokens_num={len(self.special_tokens)})"

    def __contains__(self, token: str) -> bool:
        return token in self._vocab

    def __getitem__(self, token_or_idx: Union[str, int]) -> int:
        if isinstance(token_or_idx, str):
            return self.token2id(token_or_idx)
        elif isinstance(token_or_idx, int):
            return self.id2token(token_or_idx)
        else:
            raise BadInputError(f"input type should be str or int, not {type(token_or_idx)}")

    def __iter__(self) -> Iterable['T']:
        return iter(self._vocab.items())

    def __len__(self) -> int:
        return len(self._vocab)
    
    def __add__(self, other: 'Vocabulary') -> 'Vocabulary':
        vocab = type(self).from_dict(vocab=deepcopy(self.vocab))
        for token in other.vocab.keys():
            if token in vocab:
                continue
            vocab.append(token)
        return vocab

