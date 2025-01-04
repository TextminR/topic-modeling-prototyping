"""creates a dictionary from the folder"""

import os
from dotenv import load_dotenv
from gensim import corpora
from tqdm import tqdm
import preprocessing

load_dotenv()
interactive = int(os.environ.get('INTERACTIVE', '0')) == 1

folder = input('folder: ') if interactive else os.getenv('TEXT_FOLDER', 'texts')
files = os.listdir(folder)

english = False
preprocessed = True

def generator():
    print('creating dictionary from folder: ' + folder)
    for i in tqdm(range(len(files))):
        with open(os.path.join(folder, files[i])) as file:
            if not preprocessed:
                lines = ' '.join(file.readlines())
                tokens = preprocessing.stringToTokens(lines, english)
            else:
                tokens = file.readline()
            yield tokens

dictionary = corpora.Dictionary(generator())

filename =  input('save to filename: ') if interactive else os.getenv('DICT_FILENAME', 'no-name.dict')

print('saving dictionary to: ' + filename)
dictionary.save(filename)
dictionary.save_as_text(filename + ".dbg")

