
import os
import preprocessing
from gensim import corpora
from gensim import corpora
from gensim.models import LdaModel, HdpModel, LsiModel

# lda = LdaModel.load(input('filename for lda model: '))
# dictionary = corpora.Dictionary.load(input('filename for dict: '))

lda = LdaModel.load('lda.model')
# dictionary = corpora.Dictionary.load('dictionary.dict')
dictionary = lda.id2word

lsi = LsiModel.load('lsi.model')
dictionary2 = lsi.id2word

def testStr(s):
    tokens = preprocessing.stringToTokens(s)
    print(tokens)
    print()
    bow = dictionary.doc2bow(tokens)
    bow2 = dictionary2.doc2bow(tokens)
    print(bow)
    print(bow)
    print()
    result = lda[bow]
    result2 = lsi[bow2]
    print('lda topics: ')
    for topic, prob in result:
        print(f"{prob*100}%\n", lda.print_topic(topic))
        print()
    print('lsi topics: ')
    for topic, prob in result2:
        print(f"{prob*100}%\n", lsi.print_topic(topic))
        print()



def start():
    while True:
        testStr(input('enter text: '))

start()
