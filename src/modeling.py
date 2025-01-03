import os
from corpus import FolderCorpus
import preprocessing
from gensim import corpora
from gensim.models import LdaModel, HdpModel, LsiModel

interactive = int(os.environ.get("INTERACTIVE", "0")) == 1

files = os.listdir("texts")
# files = files[:50]

english = False

dictionary = corpora.Dictionary()
print("loading dictionary")
dictionary = dictionary.load("dictionary.dict")
dictionary.filter_extremes(
    no_below=int(
        input("no below (20): ") if interactive else os.getenv("DICT_NO_BELOW", 20)
    ),
    no_above=(
        float(
            input("no above (0.5): ")
            if interactive
            else os.getenv("DICT_NO_ABOVE", 0.5)
        )
    ),
)
print(dictionary.token2id)
print("done")


c = FolderCorpus(dictionary)


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
    lda_model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        chunksize=(
            int(
                input("chunksize(2000): ")
                if interactive
                else os.getenv("LDA_CHUNKSIZE", 2000)
            )
        ),
        alpha="auto",
        eta="auto",
        iterations=(
            int(
                input("iterations(50): ")
                if interactive
                else os.getenv("LDA_ITERATIONS", 50)
            )
        ),
        num_topics=num_topics,
        passes=int(
            input("passes(20): ") if interactive else os.getenv("LDA_PASSES", 20)
        ),
        eval_every=None,
    )
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
    print("modeling lda")
    lda_model = lda_topic_modeling(c, dictionary, num_topics)
    lda_model.save("lda.model")

    print("modeling lsi")
    # hdp_model = hdp_topic_modeling(c2, dictionary)
    lsi_model = lsi_topic_modeling(c, dictionary, num_topics)
    lsi_model.save("lsi.model")

    for i, model in enumerate([lda_model, lsi_model]):
        print(f"model {i}")
        print("topics")

        for idx, topic in model.print_topics(-1):
            print(f"Topic {idx}: {topic}\n")

        print("in sample test:")
        for j, a in enumerate(c):
            print(j, files[j], model[a])

        input("next?")


model()

# print("\nHDP Topics:")
# for idx, topic in enumerate(hdp_model.print_topics(-1)[:num_topics]):
#     print(f"Topic {idx}: {topic}\n")
#
# print("\nLSI Topics:")
# for idx, topic in lsi_model.print_topics(-1):
#     print(f"Topic {idx}: {topic}\n")
#
