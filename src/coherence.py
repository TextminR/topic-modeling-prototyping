import os
from corpus import FolderCorpus
from gensim import corpora
from gensim.models import LdaModel, HdpModel, LsiModel, CoherenceModel
from dotenv import load_dotenv

load_dotenv()

interactive = int(os.environ.get("INTERACTIVE", "0")) == 1

english = False

dictionary = corpora.Dictionary.load(
    input("dictionary: ") if interactive else os.getenv("DICT_FILENAME", "")
)

c = FolderCorpus(dictionary)

ldaName = os.getenv("LDA_FILENAME", "")
lda = LdaModel.load(ldaName)

lsiName = os.getenv("LSI_FILENAME", "")
lsi = LsiModel.load(lsiName)


cmLda = CoherenceModel(model=lda, corpus=c, coherence="u_mass")
cmLsi = CoherenceModel(model=lsi, corpus=c, coherence="u_mass")

chLda = cmLda.get_coherence()
chLsi = cmLsi.get_coherence()

with open(ldaName + ".coherence", "w") as file:
    file.write(str(chLda))

with open(lsiName + ".coherence", "w") as file:
    file.write(str(chLsi))
