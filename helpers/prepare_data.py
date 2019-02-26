import gc
import sys
import random
import numpy as np
import pickle as pkl

random.seed(961)

CONSTANTS_PATH = '../Helpers/Constants'
CHARS_NUM = 50
CLASSES_NUM = 15

with open(CONSTANTS_PATH + '/ARABIC_LETTERS_LIST.pickle', 'rb') as file:
  ARABIC_LETTERS_LIST = pkl.load(file)

with open(CONSTANTS_PATH + '/DIACRITICS_LIST.pickle', 'rb') as file:
  DIACRITICS_LIST = pkl.load(file)

with open(CONSTANTS_PATH + '/CLASSES_LIST.pickle', 'rb') as file:
  CLASSES_LIST = pkl.load(file)

with open(CONSTANTS_PATH + '/CHARACTERS_MAPPING.pickle', 'rb') as file:
  CHARACTERS_MAPPING = pkl.load(file)

with open(CONSTANTS_PATH + '/CLASSES_MAPPING.pickle', 'rb') as file:
  CLASSES_MAPPING = pkl.load(file)

with open(CONSTANTS_PATH + '/REV_CLASSES_MAPPING.pickle', 'rb') as file:
  REV_CLASSES_MAPPING = pkl.load(file)

def prepare_examples_from_lines(lines):
  X = list()
  Y = list()

  for line in lines:
    for idx, ch in enumerate(line):
      if ch not in ARABIC_LETTERS_LIST:
        continue
            
      y = [0] * CLASSES_NUM
      if idx + 1 < len(line) and line[idx + 1] in DIACRITICS_LIST:
        ch_diac = line[idx + 1]
        if idx + 2 < len(line) and line[idx + 2] in DIACRITICS_LIST and ch_diac + line[idx + 2] in CLASSES_MAPPING:
          ch_diac += line[idx + 2]
        y[CLASSES_MAPPING[ch_diac]] = 1
      else:
        y[0] = 1

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
      x.append(before_need)
      x.extend([CHARACTERS_MAPPING[ch] for ch in before])
      x.extend([CHARACTERS_MAPPING[ch] for ch in after])
      x.append(after_need)

      X.append(x)
      Y.append(y)

  X = np.asarray(X)
  Y = np.asarray(Y)

  return X, Y

def prepare_data(train_file, val_file, test_file):
  train_data = list()
  with open(train_file, 'r') as file:
      train_data = file.readlines()
  print('Total number of lines (Training):', len(train_data))

  val_data = list()
  with open(val_file, 'r') as file:
      val_data = file.readlines()
  print('Total number of lines (Validation):', len(val_data))

  test_data = list()
  with open(test_file, 'r') as file:
      test_data = file.readlines()
  print('Total number of lines (Testing):', len(test_data))

  X_train, Y_train = prepare_examples_from_lines(train_data)
  with open('X_train.pickle', 'wb') as file:
    pkl.dump(X_train, file)
  with open('Y_train.pickle', 'wb') as file:
    pkl.dump(Y_train, file)
  X_train = None
  Y_train = None
  gc.collect()

  X_val, Y_val = prepare_examples_from_lines(val_data)
  with open('X_val.pickle', 'wb') as file:
    pkl.dump(X_val, file)
  with open('Y_val.pickle', 'wb') as file:
    pkl.dump(Y_val, file)
  X_val = None
  Y_val = None
  gc.collect()

  X_test, Y_test = prepare_examples_from_lines(test_data)
  with open('X_test.pickle', 'wb') as file:
    pkl.dump(X_test, file)
  with open('Y_test.pickle', 'wb') as file:
    pkl.dump(Y_test, file)
  X_test = None
  Y_test = None
  gc.collect()

if __name__ == '__main__':
  if len(sys.argv) != 4:
    sys.exit('usage: python %s [TRAINING_DATA_FILE] [VALIDATION_DATA_FILE] [TESTING_DATA_FILE]' % sys.argv[0])

  prepare_data(sys.argv[1], sys.argv[2], sys.argv[3])
