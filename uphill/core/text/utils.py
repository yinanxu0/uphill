from collections import Counter
import unicodedata
from enum import Enum, unique
# from langid import classify
from .own_langid import classify


@unique
class LANGUAGE(Enum):
    Other = -2
    Symbol = -1
    
    Number = 0
    Chinese = 1
    English = 2
    Japanese = 3
    Korean = 4

BPE_DELIMITER = b"\xe2\x96\x81".decode("utf-8")

LANGUAGE_STR_TO_ENUM = {
    'en': LANGUAGE.English, 
    'zh': LANGUAGE.Chinese, 
    'ja': LANGUAGE.Japanese, 
    'ko': LANGUAGE.Korean, 
}


def _guess_language_by_model(text):
    language_name, score = classify(text)
    if language_name in LANGUAGE_STR_TO_ENUM:
        return LANGUAGE_STR_TO_ENUM[language_name]
    return LANGUAGE.Other


def _guess_language_by_char(text):
    text = text.replace(" ", "")
    unicode_chars = [unicodedata.name(char) for char in text]
    language_names = [LANGUAGE.Other for char in text]
    
    for i in reversed(range(len(unicode_chars))) :
        if unicode_chars[i].startswith('DIGIT') :    # 1
            language_names[i] = LANGUAGE.Number    # 'DIGIT'
        
        elif (unicode_chars[i].startswith('CJK UNIFIED IDEOGRAPH') or
                unicode_chars[i].startswith('CJK COMPATIBILITY IDEOGRAPH')) :
            # 明 / 郎
            language_names[i] = LANGUAGE.Chinese    # 'CJK IDEOGRAPH'
        
        elif (unicode_chars[i].startswith('LATIN CAPITAL LETTER') or
                    unicode_chars[i].startswith('LATIN SMALL LETTER')) :
            # A / a
            language_names[i] = LANGUAGE.English    # 'LATIN LETTER'
        
        elif unicode_chars[i].startswith('HIRAGANA LETTER') :    # は こ め
            language_names[i] = LANGUAGE.Japanese    # 'GANA LETTER'
        
        elif (unicode_chars[i].startswith('AMPERSAND') or
                unicode_chars[i].startswith('APOSTROPHE') or
                unicode_chars[i].startswith('COMMERCIAL AT') or
                unicode_chars[i].startswith('DEGREE CELSIUS') or
                unicode_chars[i].startswith('EQUALS SIGN') or
                unicode_chars[i].startswith('FULL STOP') or
                unicode_chars[i].startswith('HYPHEN-MINUS') or
                unicode_chars[i].startswith('LOW LINE') or
                unicode_chars[i].startswith('NUMBER SIGN') or
                unicode_chars[i].startswith('PLUS SIGN') or
                unicode_chars[i].startswith('SEMICOLON')) :
            # & / ' / @ / ℃ / = / . / - / _ / # / + / ;
            language_names[i] = LANGUAGE.Symbol
    
    language_name, count = Counter(language_names).most_common(1)[0]
    if count*1.0/len(language_names) > 0.9:
        return language_name
    else:
        return LANGUAGE.Other


def guess_language(text):
    
    text = text.replace(BPE_DELIMITER, "")
    language = _guess_language_by_char(text)
    if language == LANGUAGE.Other:
        language = _guess_language_by_model(text)
    return language

