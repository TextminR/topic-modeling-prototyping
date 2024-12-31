"""creates a dictionary from the folder"""

import os
from gensim import corpora
from tqdm import tqdm
import preprocessing

interactive = int(os.environ.get('INTERACTIVE', '0')) == 1



files = os.listdir("texts")

english = False

def generator():
    for i in tqdm(range(len(files))):
        with open(os.path.join("texts", files[i])) as file:
            lines = ' '.join(file.readlines())
            tokens = preprocessing.stringToTokens(lines, english)
            yield tokens

dictionary = corpora.Dictionary(generator())

filename =  input('save to filename: ') if interactive else os.getenv('DICT_FILENAME', 'no-name.dict')

dictionary.save(filename)
dictionary.save_as_text(filename + ".dbg")

