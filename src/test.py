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

# Enable logging to monitor progress
logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Fetch the 20 Newsgroups dataset
# data = fetch_20newsgroups(remove=('headers', 'footers', 'quotes'))
# documents = data.data
documents = [
    "Climate change is one of the most pressing issues of our time. The rising levels of greenhouse gases in the atmosphere are causing global temperatures to increase, leading to extreme weather events such as hurricanes, droughts, and heatwaves. Reducing carbon emissions is a critical step in combating these effects. Renewable energy sources, such as solar, wind, and hydroelectric power, offer sustainable alternatives to fossil fuels. Additionally, reforestation and conservation efforts are vital for preserving biodiversity and capturing carbon dioxide from the air. Every individual has a role to play, from adopting energy-efficient practices to supporting policies that prioritize environmental protection. Together, we can make a difference.  ",
    "Artificial Intelligence (AI) is transforming industries across the globe. From healthcare to finance, AI-powered solutions are making processes more efficient and accurate. For example, machine learning algorithms can analyze vast amounts of data to predict trends or detect anomalies. However, the rapid advancement of AI raises ethical concerns. Questions about privacy, bias, and job displacement need to be addressed to ensure responsible use of technology. Despite these challenges, the potential of AI to solve complex problems and drive innovation is immense. As technology evolves, balancing progress with ethical considerations is essential.  ",
]

filters = [
    strip_punctuation,  # Remove punctuation
    strip_numeric,  # Remove numbers
    remove_stopwords,  # Remove stopwords
    strip_short,  # Remove words shorter than 3 characters
]


# Preprocessing the data
def preprocess(texts):
    processed_texts = []
    for doc in texts:
        # Tokenize and clean text
        tokens = preprocess_string(doc, filters)
        # Apply stemming to each token
        # stemmed_tokens = [stem_text(token) for token in tokens]
        stemmed_tokens = tokens
        # TODO revert

        processed_texts.append(stemmed_tokens)
    return processed_texts


print("before\n", documents[0])

processed_docs = preprocess(documents)

print("processed\n", processed_docs[0])

# Create a dictionary and corpus
dictionary = corpora.Dictionary(processed_docs)
dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=10000)
corpus = [dictionary.doc2bow(doc) for doc in processed_docs]


# Initialize topic modeling algorithms
def lda_topic_modeling(corpus, dictionary, num_topics=10):
    lda_model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=100,
        update_every=1,
        chunksize=100,
        passes=10,
        alpha="auto",
        per_word_topics=True,
    )
    return lda_model


def hdp_topic_modeling(corpus, dictionary):
    hdp_model = HdpModel(corpus=corpus, id2word=dictionary)
    return hdp_model


def lsi_topic_modeling(corpus, dictionary, num_topics=10):
    lsi_model = LsiModel(corpus=corpus, id2word=dictionary, num_topics=num_topics)
    return lsi_model


# Set number of topics (for models that need it)
num_topics = 20

# TODO tf idf step

# Create models
lda_model = lda_topic_modeling(corpus, dictionary, num_topics)
hdp_model = hdp_topic_modeling(corpus, dictionary)
lsi_model = lsi_topic_modeling(corpus, dictionary, num_topics)

# Display topics for each model
print("LDA Topics:")
for idx, topic in lda_model.print_topics(-1):
    print(f"Topic {idx}: {topic}\n")

print("\nHDP Topics:")
for idx, topic in enumerate(hdp_model.print_topics(-1)[:num_topics]):
    print(f"Topic {idx}: {topic}\n")

print("\nLSI Topics:")
for idx, topic in lsi_model.print_topics(-1):
    print(f"Topic {idx}: {topic}\n")
