
import os
from gensim import corpora
# Import libraries
from gensim import corpora
from gensim.models import LdaMulticore, HdpModel, LsiModel
from gensim.parsing.preprocessing import (
    preprocess_string,
    strip_punctuation,
    strip_numeric,
    remove_stopwords,
    strip_short,
    stem_text,
)

filename = "example-texts"


filters = [
    strip_punctuation,  # Remove punctuation
    strip_numeric,  # Remove numbers
    remove_stopwords,  # Remove stopwords
    strip_short,  # Remove words shorter than 3 characters
]

files = os.listdir("texts")
# TODO use first 50 
files = files[:50]

dictionary = corpora.Dictionary()
print("loading dictionary")
dictionary = dictionary.load('dictionary.dict')
print("done")


class FolderCorpus:
    def __init__(self) -> None:
        pass

    def __iter__(self):
        for i in range(len(files)):
            with open(os.path.join("texts", files[i])) as file:
                lines = ' '.join(file.readlines())
                tokens = preprocess_string(lines, filters)
                # Apply stemming to each token
                stemmed_tokens = [stem_text(token) for token in tokens]
                # print(stemmed_tokens, "stemmed")
                yield dictionary.doc2bow(stemmed_tokens)



c = FolderCorpus()




# Enable logging to monitor progress
# logging.basicConfig(
#     format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
# )
# warnings.filterwarnings("ignore", category=DeprecationWarning)



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
    lda_model = LdaMulticore(corpus, id2word=dictionary, num_topics=num_topics, passes=10)
    return lda_model


def hdp_topic_modeling(corpus, dictionary):
    hdp_model = HdpModel(corpus=corpus, id2word=dictionary)
    return hdp_model


def lsi_topic_modeling(corpus, dictionary, num_topics=10):
    lsi_model = LsiModel(corpus=corpus, id2word=dictionary, num_topics=num_topics)
    return lsi_model


# Set number of topics (for models that need it)
num_topics = 50

# TODO tf idf step


# Create models
def model():
    global lda_model
    global lsi_model
    num_topics = int(input("enter num topics: "))
    print('modeling lda')
    lda_model = lda_topic_modeling(c, dictionary, num_topics)
    lda_model.save('lda.model')
    
    print('modeling lsi')
    # hdp_model = hdp_topic_modeling(c2, dictionary)
    lsi_model = lsi_topic_modeling(c, dictionary, num_topics)
    lsi_model.save('lsi.model')

    for i, model in enumerate([lda_model, lsi_model]):
        print(f"model {i}")
        print("topics")

        for idx, topic in model.print_topics(-1):
            print(f"Topic {idx}: {topic}\n")

        print("in sample test:")
        textFile = open(filename)
        for _, a in enumerate(c):
            print(textFile.readline())
            print(model[a], "\n")

        input('next?')


model()

# print("\nHDP Topics:")
# for idx, topic in enumerate(hdp_model.print_topics(-1)[:num_topics]):
#     print(f"Topic {idx}: {topic}\n")
#
# print("\nLSI Topics:")
# for idx, topic in lsi_model.print_topics(-1):
#     print(f"Topic {idx}: {topic}\n")
#
