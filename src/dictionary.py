"""creates a dictionary from the folder"""

import os
from gensim import corpora
# Import libraries
from gensim import corpora
from gensim.models import LdaModel, HdpModel, LsiModel
from gensim.parsing.preprocessing import (
    preprocess_string,
    strip_punctuation,
    strip_numeric,
    remove_stopwords,
    strip_short,
    stem_text,
)
from tqdm import tqdm
from nltk.stem.snowball import SnowballStemmer

import nltk


filters = [
    strip_punctuation,  # Remove punctuation
    strip_numeric,  # Remove numbers
    remove_stopwords,  # Remove stopwords
    strip_short,  # Remove words shorter than 3 characters
]


nltk.download('stopwords')
germanStopwords = nltk.corpus.stopwords.words('german')
snowball = SnowballStemmer("german")

dictionary = corpora.Dictionary()

files = os.listdir("texts")
files = files[:1]

english = False

for i in tqdm(range(len(files))):
    with open(os.path.join("texts", files[i])) as file:
        lines = ' '.join(file.readlines())

        tokens = preprocess_string(lines, filters)
        # Apply stemming to each token
        stemmed_tokens = [stem_text(token) for token in tokens] if english else [snowball.stem(token) for token in tokens]
        if not english:
            print('stemming german')
            stemmed_tokens = [w for w in stemmed_tokens if w not in germanStopwords]
        print(stemmed_tokens)
        dictionary.add_documents([stemmed_tokens])

dictionary.save(input('save to filename: '))

