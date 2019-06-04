# -*- coding: utf-8 -*-

import argparse
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

CONSTANTS_PATH = 'constants'
CLASSES_NAMES = ['No Diacritic', 'Fatha', 'Fathatan', 'Damma', 'Dammatan', 'Kasra', 'Kasratan', 'Sukun', 'Shadda', 'Shadda + Fatha', 'Shadda + Fathatan', 'Shadda + Damma', 'Shadda + Dammatan', 'Shadda + Kasra', 'Shadda + Kasratan']

def get_diacritic_class(idx, line, arabic_letters, diacritic_classes):
  if idx + 1 >= len(line) or line[idx + 1] not in diacritic_classes:
    # No diacritic
    return CLASSES_NAMES[0]

  diac = line[idx + 1]

  if idx + 2 >= len(line) or line[idx + 2] not in diacritic_classes:
    # Only one diacritic
    return CLASSES_NAMES[diacritic_classes.index(diac) + 1]

  diac += line[idx + 2]

  try:
    # Try the possibility of double diacritics
    return CLASSES_NAMES[diacritic_classes.index(diac) + 1]
  except:
    try:
      # Try the possibility of reversed double diacritics
      return CLASSES_NAMES[diacritic_classes.index(diac[::-1]) + 1]
    except:
      # Otherwise consider only the first diacritic
      return CLASSES_NAMES[diacritic_classes.index(diac[0]) + 1]

def get_diacritics_classes(line, arabic_letters, diacritic_classes):
  classes = list()
  for idx, char in enumerate(line):
    if char in arabic_letters:
      classes.append(get_diacritic_class(idx, line, arabic_letters, diacritic_classes))
  return classes

def plot_confusion_matrix(y_true, y_pred, subplot,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
  """
  This function prints and plots the confusion matrix.
  Normalization can be applied by setting `normalize=True`.
  """
  if not title:
      if normalize:
          title = 'Normalized confusion matrix'
      else:
          title = 'Confusion matrix, without normalization'

  # Only use the labels that appear in the data
  classes = unique_labels(y_true, y_pred)
  new_classes = list()
  for class_name in CLASSES_NAMES:
    if class_name in classes:
      new_classes.append(class_name)
  classes = new_classes
  # Compute confusion matrix
  cm = confusion_matrix(y_true, y_pred, labels=classes)
  if normalize:
      cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
      print("Normalized confusion matrix")
  else:
      print('Confusion matrix, without normalization')

  print(cm)

  ax = plt.subplot(1, 2, subplot)
  im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
  ax.figure.colorbar(im, ax=ax)
  # We want to show all ticks...
  ax.set(xticks=np.arange(cm.shape[1]),
         yticks=np.arange(cm.shape[0]),
         # ... and label them with the respective list entries
         xticklabels=classes, yticklabels=classes,
         title=title,
         ylabel='True label',
         xlabel='Predicted label')

  # Rotate the tick labels and set their alignment.
  plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
           rotation_mode="anchor")

  # Loop over data dimensions and create text annotations.
  fmt = '.2f' if normalize else 'd'
  thresh = cm.max() / 2.
  for i in range(cm.shape[0]):
      for j in range(cm.shape[1]):
          ax.text(j, i, format(cm[i, j], fmt),
                  ha="center", va="center",
                  color="white" if cm[i, j] > thresh else "black")
  return ax

if __name__ =='__main__':
  parser = argparse.ArgumentParser(description='Builds DER Figure')
  parser.add_argument('-ofp', '--original-file-path', help='File path to original text', required=True)
  parser.add_argument('-stfp', '--small-target-file-path', help='File path to target text', required=True)
  parser.add_argument('-btfp', '--big-target-file-path', help='File path to target text', required=True)
  args = parser.parse_args()

  with open(CONSTANTS_PATH + '/ARABIC_LETTERS_LIST.pickle', 'rb') as file:
    ARABIC_LETTERS_LIST = pkl.load(file)

  with open(CONSTANTS_PATH + '/FFNN_CLASSES_MAPPING.pickle', 'rb') as file:
    CLASSES_LIST = list(pkl.load(file))

  with open(args.original_file_path, 'r') as file:
    original_content = file.readlines()

  with open(args.small_target_file_path, 'r') as file:
    small_target_content = file.readlines()

  with open(args.big_target_file_path, 'r') as file:
    big_target_content = file.readlines()

  original_classes = list()
  small_target_classes = list()
  big_target_classes = list()

  for original_line, small_target_line, big_target_line in zip(original_content, small_target_content, big_target_content):
    original_classes.extend(get_diacritics_classes(original_line, ARABIC_LETTERS_LIST, CLASSES_LIST))
    small_target_classes.extend(get_diacritics_classes(small_target_line, ARABIC_LETTERS_LIST, CLASSES_LIST))
    big_target_classes.extend(get_diacritics_classes(big_target_line, ARABIC_LETTERS_LIST, CLASSES_LIST))

  plot_confusion_matrix(original_classes, small_target_classes, 1, True, 'Recurrent normalized model confusion matrix - without extra train')
  plot_confusion_matrix(original_classes, big_target_classes, 2, True, 'Recurrent normalized model confusion matrix - with extra train')

  plt.show()
