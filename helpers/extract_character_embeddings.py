import argparse
import pickle as pkl

from keras.models import load_model
from keras.initializers import glorot_normal
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from optimizer import NormalizedOptimizer

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--model-path', required=True)
  args = parser.parse_args()

  custom_objects = {
    'GlorotNormal': glorot_normal(),
    'CRF': CRF,
    'crf_loss': crf_loss,
    'NormalizedOptimizer': NormalizedOptimizer
  }
  model = load_model(args.model_path, custom_objects=custom_objects)

  model.summary()

  embeddings = model.layers[1].get_weights()[0]

  with open('embeddings.pkl', 'wb') as file:
    pkl.dump(embeddings, file)
