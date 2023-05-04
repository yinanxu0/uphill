import os
import pytest
from pathlib import Path


from uphill.core.text import (
    LANGUAGE, guess_language,
    Tokenizer, Vocabulary
)


test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


@pytest.mark.parametrize(
    'text', [
        "From the frozen snows of the polar regions to greece", 
        "The tales are adapted to the needs of british children by various hands", 
        "<noise> from new caledonia <noise> to zululand", 
    ]
)
@pytest.mark.parametrize(
    'bpe_model', [
        os.path.join(test_dir, "toydata/models/bpe.model"), 
        None, 
    ]
)
@pytest.mark.parametrize(
    'special_tokens', [
        ["<noise>"], 
        None, 
    ]
)
def test_english_tokenize_and_detokenize(text, bpe_model, special_tokens):
    tokenizer = Tokenizer(bpe_model=bpe_model, special_tokens=special_tokens)
    
    tokens = tokenizer.tokenize(text=text)
    text_recover = tokenizer.detokenize(tokens=tokens)
    while "  " in text_recover:
        text_recover  = text_recover.replace("  ", " ")
    assert text_recover == text.lower()


@pytest.mark.parametrize(
    'text', [
        "比赛就此定格在<noise>第一回合二分钟",
        "而据上海易居研究院提供的市场成交报告",
    ]
)
@pytest.mark.parametrize(
    'bpe_model', [
        os.path.join(test_dir, "toydata/models/bpe.model"), 
        None, 
    ]
)
@pytest.mark.parametrize(
    'special_tokens', [
        ["<noise>"], 
        None, 
    ]
)
def test_chinese_tokenize_and_detokenize(text, bpe_model, special_tokens):
    tokenizer = Tokenizer(bpe_model=bpe_model, special_tokens=special_tokens)
    
    tokens = tokenizer.tokenize(text=text)
    text_recover = tokenizer.detokenize(tokens=tokens)
    text = text.replace("<noise>", " <noise> ")

    assert text == text_recover


@pytest.mark.parametrize(
    'text', [
        "回忆,你真是精彩 bye bye never say goodbye 未来",
        "而据上海易居研究院<noise>提供的市场成交报告",
        "回忆你真是精彩<noise> bye bye never say goodbye 未来",
    ]
)
@pytest.mark.parametrize(
    'bpe_model', [
        os.path.join(test_dir, "toydata/models/bpe.model"), 
        None, 
    ]
)
@pytest.mark.parametrize(
    'special_tokens', [
        ["<noise>"], 
        None, 
    ]
)
def test_english_chinese_tokenize_and_detokenize(text, bpe_model, special_tokens):
    tokenizer = Tokenizer(bpe_model=bpe_model, special_tokens=special_tokens)
    
    tokens = tokenizer.tokenize(text=text)
    text_recover = tokenizer.detokenize(tokens=tokens)
    
    text = text.replace("<noise>", " <noise> ")
    text = text.replace(",", " , ")
    while "  " in text:
        text  = text.replace("  ", " ")
        
    text_recover = text_recover.replace(",", " , ")
    while "  " in text_recover:
        text_recover  = text_recover.replace("  ", " ")
    assert text == text_recover


@pytest.mark.parametrize(
    'vocab_path', [
        os.path.join(test_dir, "toydata/text/words.txt"),
    ]
)
def test_vocab(vocab_path):
    vocab = Vocabulary.from_file(vocab_path=vocab_path)
    assert len(vocab) == 5029

    assert '<noise>' in vocab
    assert vocab['<noise>'] == 3
    assert vocab[3] == '<noise>'
    ## add existed token, length keeps constant
    vocab.append('<noise>')
    assert len(vocab) == 5029
    
    vocab.append('<yinanxu_append>')
    assert '<yinanxu_append>' in vocab
    assert len(vocab) == 5030
    assert vocab['<yinanxu_append>'] == 5029
    assert vocab[5029] == '<yinanxu_append>'
    
    vocab.insert(20, '<yinanxu_insert>')
    assert '<yinanxu_insert>' in vocab
    assert len(vocab) == 5031
    assert vocab['<yinanxu_insert>'] == 20
    assert vocab[20] == '<yinanxu_insert>'
    # index of '<yinanxu_append>' should increased by 1
    assert vocab['<yinanxu_append>'] == 5030
    assert vocab[5030] == '<yinanxu_append>'
    
    vocab.remove('<yinanxu_append>')
    assert '<yinanxu_append>' not in vocab
    assert len(vocab) == 5030
    # index of '<yinanxu_insert>' should keep constant
    assert vocab['<yinanxu_insert>'] == 20
    assert vocab[20] == '<yinanxu_insert>'

