import re
from unidecode import unidecode
from rank_bm25 import BM25Okapi

_whitespace_re = re.compile(r'\s+')

# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = [(re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in [
    ('mrs', 'misess'),
    ('mr', 'mister'),
    ('dr', 'doctor'),
    ('st', 'saint'),
    ('co', 'company'),
    ('jr', 'junior'),
    ('maj', 'major'),
    ('gen', 'general'),
    ('drs', 'doctors'),
    ('rev', 'reverend'),
    ('lt', 'lieutenant'),
    ('hon', 'honorable'),
    ('sgt', 'sergeant'),
    ('capt', 'captain'),
    ('esq', 'esquire'),
    ('ltd', 'limited'),
    ('col', 'colonel'),
    ('ft', 'fort'),
]]

def expand_abbreviations(text):
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text

def lowercase(text):
    return text.lower()

def collapse_whitespace(text):
    return re.sub(_whitespace_re, ' ', text)

def convert_to_ascii(text):
    return unidecode(text)

def basic_cleaners(text):
    '''Basic pipeline that lowercases and collapses whitespace without transliteration.'''
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text

def english_cleaners(text):
    '''Pipeline for English text, including number and abbreviation expansion.'''
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = expand_abbreviations(text)
    # text = collapse_whitespace(text)
    return text

# f = open("source/reference.txt", "r").read().split('\n\n')
# tokenized_corpus = [doc.split(" ") for doc in f]
# bm25 = BM25Okapi(tokenized_corpus)

# def myBM25(user_text):
#     tokenized_query = user_text.split(" ")
#     contexts = bm25.get_top_n(tokenized_query, tokenized_corpus, n=3)
#     prompt = "Answer the question below, you can refer to but NOT limited to the contexts \n\nContext: \n"
#     for context in contexts:
#         prompt += ' '.join(context) + "\n\n###\n\n"
#     prompt += "---\n\nQuestion: " + user_text + "\nAnswer:"
#     return prompt



