import argparse
import numpy as np

from keras.models import load_model
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from optimizer import NormalizedOptimizer
from os import sep, listdir
from os.path import isfile, join

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Builds Averaged Models')
  parser.add_argument('-in', '--folder-path', required=True)
  parser.add_argument('-avg', '--average-list', nargs='+', type=int, default=[5, 10, 20])
  args = parser.parse_args()

  checkpoints = [join(args.folder_path, file) for file in listdir(args.folder_path) if isfile(join(args.folder_path, file)) if '.ckpt' in file]
  checkpoints = list(sorted(checkpoints, reverse=True))

  models = list()
  for checkpoint in checkpoints:
    print('Loading checkpoint: %s' % checkpoint)
    models.append(load_model(checkpoint, custom_objects={'CRF': CRF, 'crf_loss': crf_loss, 'NormalizedOptimizer': NormalizedOptimizer}))

  for avg in args.average_list:
    print('Building model with %s averaged checkpoint' % avg)
    
    avg_models = models[:avg]
    weights = [model.get_weights() for model in avg_models]

    new_weights = list()
    for weights_list_tuple in zip(*weights):
      new_weights.append([np.array(weights_).mean(axis=0) for weights_ in zip(*weights_list_tuple)])

    avg_model = avg_models[0]
    avg_model.set_weights(new_weights)

    print('Saving averaged model to: %s' % join(args.folder_path, 'avg_%s' % avg))
    avg_model.save(join(args.folder_path, 'avg_%s.ckpt' % avg))
