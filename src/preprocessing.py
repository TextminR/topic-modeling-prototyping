import os
from gensim.parsing.preprocessing import (
    preprocess_string,
    strip_punctuation,
    strip_numeric,
    remove_stopwords,
    strip_short,
    stem_text,
    strip_tags,
    strip_multiple_whitespaces,
    strip_non_alphanum
)
from tqdm import tqdm
from nltk.stem.snowball import SnowballStemmer
import nltk

filters = [
    strip_multiple_whitespaces,
    # strip_punctuation,  # Remove punctuation
    strip_non_alphanum,
    strip_numeric,  # Remove numbers
    remove_stopwords,  # Remove stopwords
    strip_tags,
    strip_short,  # Remove words shorter than 3 characters
]
nltk.download("stopwords")
germanStopwords = nltk.corpus.stopwords.words("german")
snowball = SnowballStemmer("german")


def stringToTokens(text, english=False):
    # tokenization + apply filters
    tokens = preprocess_string(text, filters)
    # stemming
    stemmed_tokens = (
        [stem_text(token) for token in tokens]
        if english
        else [snowball.stem(token) for token in tokens]
    )
    # german stopwords
    if not english:
        stemmed_tokens = [w for w in stemmed_tokens if w not in germanStopwords]
    return stemmed_tokens


# TODO at least document trying n grams

# additional src: https://nickyreinert.de/blog/2020/12/09/einf%C3%BChrung-in-stemming-und-lemmatisierung-deutscher-texte-mit-python/


def convertFolder(folder="texts", out="preprocessed_texts"):
    files = os.listdir(folder)
    os.makedirs(out, exist_ok=True)
    for fileName in tqdm(files):
        with open(os.path.join(folder, fileName)) as fileIn:
            lines = " ".join(fileIn.readlines())
            tokens = stringToTokens(lines)
            outString = " ".join(tokens)
            with open(os.path.join(out, fileName), "w") as fileOut:
                fileOut.write(outString)


def iterateFolder(name):
    files = os.listdir(name)
    # for fn in tqdm(files):
    for fn in files:
        with open(os.path.join(name, fn)) as file:
            file.readlines()


if __name__ == "__main__":
    name = input("folder: ")
    for i in range(10):
        iterateFolder(name)
        print(f"fin {i}")
