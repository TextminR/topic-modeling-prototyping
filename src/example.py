from gensim import corpora

# Import libraries
from sklearn.datasets import fetch_20newsgroups
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
import logging
import warnings

filename = "example-texts"


documents = [
    "Human machine interface for lab abc computer applications",
    "A survey of user opinion of computer system response time",
    "The EPS user interface management system",
    "System and human system engineering testing of EPS",
    "Relation of user perceived response time to error measurement",
    "The generation of random binary unordered trees",
    "The intersection graph of paths in trees",
    "Graph minors IV Widths of trees and well quasi ordering",
    "Graph minors A survey",
]


class MyCorpus:
    def __init__(self) -> None:
        self.i = 0

    def __iter__(self):
        j = 0
        for line in documents:
            self.i += 1
            j += 1
            yield (line, self.i, j)


filters = [
    strip_punctuation,  # Remove punctuation
    strip_numeric,  # Remove numbers
    remove_stopwords,  # Remove stopwords
    strip_short,  # Remove words shorter than 3 characters
]


dictionary = corpora.Dictionary()


class ExampleCorpus:
    def __init__(self) -> None:
        pass

    def __iter__(self):
        for line in open(filename):
            tokens = preprocess_string(line, filters)
            # Apply stemming to each token
            stemmed_tokens = [stem_text(token) for token in tokens]
            # print(stemmed_tokens, "stemmed")
            dictionary.add_documents([stemmed_tokens])
            # TODO fix
            yield dictionary.doc2bow(stemmed_tokens)


c = MyCorpus()


def it():
    for a in c:
        print(a)


# dictionary = corpora.Dictionary(line.lower().split() for line in c)

for line in open(filename):
    tokens = preprocess_string(line, filters)
    # Apply stemming to each token
    stemmed_tokens = [stem_text(token) for token in tokens]
    dictionary.add_documents([stemmed_tokens])

c2 = ExampleCorpus()


def it2():
    for a in c2:
        print(a)


# Enable logging to monitor progress
# logging.basicConfig(
#     format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
# )
# warnings.filterwarnings("ignore", category=DeprecationWarning)

processed_docs = c2


# Create a dictionary and corpus
# dictionary.filter_extremes(no_below=0, no_above=0.5, keep_n=10000)
# corpus = [dictionary.doc2bow(doc) for doc in processed_docs]


lda_model = None
lsi_model = None


# Initialize topic modeling algorithms
def lda_topic_modeling(corpus, dictionary, num_topics=10):
    # lda_model = LdaModel(
    #     corpus=corpus,
    #     id2word=dictionary,
    #     num_topics=num_topics,
    #     random_state=100,
    #     update_every=1,
    #     chunksize=100,
    #     passes=10,
    #     alpha="auto",
    #     per_word_topics=True,
    # )
    lda_model = LdaModel(corpus, id2word=dictionary, num_topics=num_topics, passes=10)
    return lda_model


def hdp_topic_modeling(corpus, dictionary):
    hdp_model = HdpModel(corpus=corpus, id2word=dictionary)
    return hdp_model


def lsi_topic_modeling(corpus, dictionary, num_topics=10):
    lsi_model = LsiModel(corpus=corpus, id2word=dictionary, num_topics=num_topics)
    return lsi_model


# Set number of topics (for models that need it)
num_topics = 2

# TODO tf idf step


# Create models
# it2()
def model():
    global lda_model
    global lsi_model
    num_topics = int(input("enter num topics: "))
    lda_model = lda_topic_modeling(c2, dictionary, num_topics)
    # hdp_model = hdp_topic_modeling(c2, dictionary)
    lsi_model = lsi_topic_modeling(c2, dictionary, num_topics)

    for i, model in enumerate([lda_model, lsi_model]):
        print(f"model {i}")
        print("topics")

        for idx, topic in model.print_topics(-1):
            print(f"Topic {idx}: {topic}\n")

        print("in sample test:")
        textFile = open(filename)
        for _, a in enumerate(c2):
            print(textFile.readline())
            print(model[a], "\n")


model()

# print("\nHDP Topics:")
# for idx, topic in enumerate(hdp_model.print_topics(-1)[:num_topics]):
#     print(f"Topic {idx}: {topic}\n")
#
# print("\nLSI Topics:")
# for idx, topic in lsi_model.print_topics(-1):
#     print(f"Topic {idx}: {topic}\n")
#
