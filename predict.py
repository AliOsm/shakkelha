import argparse

import numpy as np
import pickle as pkl
import tensorflow as tf

from os import listdir, environ
from os.path import isdir, isfile, join
from helpers.optimizer import NormalizedOptimizer
from keras.models import load_model as keras_load_model
from keras.initializers import glorot_normal
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from tensorflow import keras
from tensorflow.keras.models import load_model as tf_load_model

CONSTANTS_PATH = 'helpers/constants'

def get_model_path(model_type, model_number, model_size, model_average):
  if model_type == 'rnn' and model_size == None:
    raise ValueError('model_size should be specified for rnn models')
  if model_type == 'rnn' and model_average == None:
    raise ValueError('model_average should be specified for rnn models')

  model_path = 'models/%s_models' % model_type
  model_path = join(model_path, [folder for folder in listdir(model_path) if isdir(join(model_path, folder)) if str(model_number) == folder[0]][0])

  if model_type == 'ffnn':
    model_path = join(model_path, 'model.ckpt')
  elif model_type == 'rnn':
    if tf.test.is_gpu_available():
      model_path = join(model_path, model_size + '_data', 'avg_%s.ckpt' % model_average)
    else:
      model_path = join(model_path, model_size + '_data', 'lstm', 'avg_%s.ckpt' % model_average)

  return model_path

def read_data(file_path):
  with open(file_path, 'r') as file:
    data = file.readlines()
  data = [line.strip() for line in data]
  return data

def write_data(data, file_path):
  with open(file_path, 'w') as file:
    file.write('\n'.join(data))

def remove_diacritics(data, DIACRITICS_LIST):
  return data.translate(str.maketrans('', '', ''.join(DIACRITICS_LIST)))

def predict_ffnn(line, model, ARABIC_LETTERS_LIST, DIACRITICS_LIST, CHARACTERS_MAPPING, REV_CLASSES_MAPPING):
  CHARS_NUM = 50

  output = ''
  for idx, char in enumerate(line):
    if char in DIACRITICS_LIST:
      continue

    output += char

    if char not in ARABIC_LETTERS_LIST:
      continue

    before = list()
    after = list()

    for idxb in range(idx - 1, -1, -1):
      if len(before) >= CHARS_NUM:
        break
      if line[idxb] not in DIACRITICS_LIST:
        before.append(line[idxb])
    before = before[::-1]
    before_need = CHARS_NUM - len(before)

    for idxa in range(idx, len(line)):
      if len(after) >= CHARS_NUM:
        break
      if line[idxa] not in DIACRITICS_LIST:
        after.append(line[idxa])
    after_need = CHARS_NUM - len(after)

    x = list()
    x.extend([1] * before_need)
    x.extend([CHARACTERS_MAPPING[ch] if ch in CHARACTERS_MAPPING else 0 for ch in before])
    x.extend([CHARACTERS_MAPPING[ch] if ch in CHARACTERS_MAPPING else 0 for ch in after])
    x.extend([1] * after_need)
    x = np.asarray(x).reshape(1, -1)

    pred = np.argmax(model.predict(x))

    if pred == 0:
      continue

    output += REV_CLASSES_MAPPING[pred]

  return output

def predict_rnn(line, model, ARABIC_LETTERS_LIST, DIACRITICS_LIST, CHARACTERS_MAPPING, REV_CLASSES_MAPPING):
  x = [CHARACTERS_MAPPING['<SOS>']]
  for idx, char in enumerate(line):
    if char in DIACRITICS_LIST:
      continue
    x.append(CHARACTERS_MAPPING[char])
  x.append(CHARACTERS_MAPPING['<EOS>'])
  x = np.array(x).reshape(1, -1)

  predictions = model.predict(x).squeeze()
  predictions = predictions[1:]
  
  output = ''
  for char, prediction in zip(remove_diacritics(line, DIACRITICS_LIST), predictions):
    output += char
    
    if char not in ARABIC_LETTERS_LIST:
      continue
    
    prediction = np.argmax(prediction)

    if '<' in REV_CLASSES_MAPPING[prediction]:
      continue

    output += REV_CLASSES_MAPPING[prediction]

  return output

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Diacritize Given File Using Specific Model')
  parser.add_argument('-t', '--model-type', required=True, choices=['ffnn', 'rnn'])
  parser.add_argument('-n', '--model-number', required=True)
  parser.add_argument('-s', '--model-size', default='small', choices=['small', 'big'])
  parser.add_argument('-a', '--model-average', choices=[1, 5, 10, 20], type=int)
  parser.add_argument('-in', '--input-file-path', required=True)
  parser.add_argument('-out', '--output-file-path', required=True)
  args = parser.parse_args()

  args.model_type = args.model_type.lower()
  args.model_size = args.model_size.lower()

  # set defaults for FFNN models
  if args.model_type == 'ffnn':
    args.model_size = 'small'

  # shut up tensorflow ans keras
  environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
  tf.logging.set_verbosity(tf.logging.ERROR)

  # load the model
  model_path = get_model_path(args.model_type, args.model_number, args.model_size, args.model_average)
  print('Loading model from: %s' % model_path)
  custom_objects = {
    'GlorotNormal': glorot_normal(),
    'CRF': CRF,
    'crf_loss': crf_loss,
    'NormalizedOptimizer': NormalizedOptimizer
  }
  # try to load the model using keras load model method
  try:
    model = keras_load_model(model_path, custom_objects=custom_objects)
  # if error occurred, try to load using tensorflow.keras load model method
  except:
    model = tf_load_model(model_path)
  model.summary()
  print('Model loaded successfully!')

  # load the data
  print('Loading data...')
  data = read_data(args.input_file_path)
  print('%s lines loaded' % len(data))

  # load the needed constants
  print('Loading constants...')
  # add missing character from the library
  with open(join(CONSTANTS_PATH, 'ARABIC_LETTERS_LIST.pickle'), 'rb') as file:
    ARABIC_LETTERS_LIST = pkl.load(file)
  with open(join(CONSTANTS_PATH, 'DIACRITICS_LIST.pickle'), 'rb') as file:
    DIACRITICS_LIST = pkl.load(file)
  with open(join(CONSTANTS_PATH, '%s_%s_CHARACTERS_MAPPING.pickle' % (args.model_type.upper(), args.model_size.upper())), 'rb') as file:
    CHARACTERS_MAPPING = pkl.load(file)
  with open(join(CONSTANTS_PATH, '%s_REV_CLASSES_MAPPING.pickle' % args.model_type.upper()), 'rb') as file:
    REV_CLASSES_MAPPING = pkl.load(file)

  # start predicting
  print('Start predicting...')
  outputs = list()
  for idx, line in enumerate(data):
    if args.model_type == 'ffnn':
      outputs.append(predict_ffnn(line, model, ARABIC_LETTERS_LIST,
                                               DIACRITICS_LIST,
                                               CHARACTERS_MAPPING,
                                               REV_CLASSES_MAPPING))
    elif args.model_type == 'rnn':
      outputs.append(predict_rnn(line, model, ARABIC_LETTERS_LIST,
                                              DIACRITICS_LIST,
                                              CHARACTERS_MAPPING,
                                              REV_CLASSES_MAPPING))
    print('%s/%s (%0.2f)' % (idx + 1, len(data), (idx + 1) / len(data) * 100), end='\r')
  print('')

  # write predictions
  print('Write predictions...')
  write_data(outputs, args.output_file_path)
