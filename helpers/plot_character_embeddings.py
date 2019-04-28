import argparse
import pickle as pkl
import matplotlib.pyplot as plt

from os import sep, listdir
from os.path import isdir, join
from sklearn.manifold import TSNE

def tsne_plot(tokens, labels):
  tsne_model = TSNE(perplexity=20, random_state=1, n_components=2, init='pca')
  new_values = tsne_model.fit_transform(tokens)

  x = []
  y = []
  for value in new_values:
    x.append(value[0])
    y.append(value[1])

  plt.figure(figsize=(16, 8)) 
  for i in range(len(x)):
    plt.scatter(x[i], y[i])
    plt.annotate(labels[i],
                 xy=(x[i], y[i]),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')
  plt.tight_layout()
  plt.show()

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Plots Character Embeddings in 2D Space')
  parser.add_argument('-emb', '--embeddings-path', required=True)
  parser.add_argument('-vocab', '--vocab-path', required=True)
  args = parser.parse_args()

  with open(args.embeddings_path, 'rb') as file:
    embeddings = pkl.load(file)

  with open(args.vocab_path, 'rb') as file:
    vocab = pkl.load(file)

  tsne_plot(embeddings, vocab)
