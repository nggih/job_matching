import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize

# corpus
articles = pd.read_csv('./data/corpus/data.csv')
print(articles.head())

data = list(articles['data'])
tagged_data = [TaggedDocument(words = word_tokenize(_d.lower()), tags = [str(i)]) for i, _d in enumerate(data)]
model = Doc2Vec(vector_size = 100,
min_count = 10,
epochs = 50
)
model.build_vocab(tagged_data)
model.train(tagged_data,
total_examples = model.corpus_count,
epochs = model.epochs)
model.save('doc2vec.model')
print("Model saved")