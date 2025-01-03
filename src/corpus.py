import os
from gensim import corpora

interactive = int(os.environ.get("INTERACTIVE", "0")) == 1

files = os.listdir("processed_texts")

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


class FolderCorpus:
    def __init__(self, dictionary) -> None:
        self.dictionary = dictionary

    def __iter__(self):
        for i in range(len(files)):
            with open(os.path.join("texts", files[i])) as file:
                tokens = file.readline().split(' ')

                yield self.dictionary.doc2bow(tokens)
