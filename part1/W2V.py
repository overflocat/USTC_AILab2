import glob
import gensim
from gensim import corpora
from gensim.models import word2vec
import logging
import string

def getCorpus():
    documents = []
    txtNames = glob.glob("original/*.txt")
    for fileName in txtNames:
        fp = open(fileName)
        buf = fp.readline()
        documents.append(buf)

    stoplist = set('for a of the and to in at'.split())
    texts = [[word for word in document.translate(string.maketrans("", ""), string.punctuation).lower().split() if word not in stoplist]
             for document in documents]

    #Actually dictionary and corpus are of no use here
    dictionary = corpora.Dictionary(texts)
    dictionary.filter_extremes(no_below=10, no_above=0.7, keep_n=50000)
    dictionary.save('tmp/imdb.dict')

    corpus = [dictionary.doc2bow(text) for text in texts]
    corpora.MmCorpus.serialize('tmp/imdb.mm', corpus)

    return texts

def TrainW2V(texts):
    model = word2vec.Word2Vec(texts, size=150, workers=4)
    model.save("med150.model.bin")

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
texts = getCorpus()
TrainW2V(texts)