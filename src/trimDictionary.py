import os
from dotenv import load_dotenv
from gensim import corpora
from tqdm import tqdm
import preprocessing


load_dotenv()

interactive = int(os.environ.get("INTERACTIVE", "0")) == 1

dictionary = corpora.Dictionary.load(
    input("dictionary: ") if interactive else os.getenv("OLD_DICT_FILENAME", "")
)
dictionary.filter_extremes(
    no_below=int(input("no below: ") if interactive else os.getenv("DICT_NO_BELOW", 1)),
    no_above=(
        float(input("no above: ") if interactive else os.getenv("DICT_NO_ABOVE", 1))
    ),
)

filename = input("save to: ") if interactive else os.getenv("DICT_FILENAME", "")
dictionary.save(filename)
dictionary.save_as_text(filename + ".dbg")
