# -*- coding: utf-8 -*-

import argparse
import pickle as pkl
import matplotlib.pyplot as plt
import arabic_reshaper

from bidi.algorithm import get_display
from matplotlib.pyplot import figure

CONSTANTS_PATH = 'constants'

def remove_diacritics(content, diacritic_classes):
    return content.translate(str.maketrans('', '', ''.join(diacritic_classes)))

def get_diacritic_class(idx, line, arabic_letters, diacritic_classes):
  if idx + 1 >= len(line) or line[idx + 1] not in diacritic_classes:
    # No diacritic
    return 0

  diac = line[idx + 1]

  if idx + 2 >= len(line) or line[idx + 2] not in diacritic_classes:
    # Only one diacritic
    return diacritic_classes.index(diac) + 1

  diac += line[idx + 2]

  try:
    # Try the possibility of double diacritics
    return diacritic_classes.index(diac) + 1
  except:
    try:
      # Try the possibility of reversed double diacritics
      return diacritic_classes.index(diac[::-1]) + 1
    except:
      # Otherwise consider only the first diacritic
      return diacritic_classes.index(diac[0]) + 1

def get_diacritics_classes(line, arabic_letters, diacritic_classes):
  classes = list()
  for idx, char in enumerate(line):
    if char in arabic_letters:
      classes.append(get_diacritic_class(idx, line, arabic_letters, diacritic_classes))
  return classes

def clear_line(line, arabic_letters, diacritic_classes):
  return ' '.join(''.join([char if char in list(arabic_letters) + diacritic_classes + [' '] else ' ' for char in line]).split())

def count_error_frequency(original_file, target_file, arabic_letters, diacritic_classes):
  with open(original_file, 'r') as file:
    original_content = file.readlines()

  with open(target_file, 'r') as file:
    target_content = file.readlines()

  assert(len(original_content) == len(target_content))

  freq = dict()

  for idx, (original_line, target_line) in enumerate(zip(original_content, target_content)):
    original_line = clear_line(original_line, arabic_letters, diacritic_classes)
    target_line = clear_line(target_line, arabic_letters, diacritic_classes)

    original_line = original_line.split()
    target_line = target_line.split()

    assert(len(original_line) == len(target_line))

    for (original_word, target_word) in zip(original_line, target_line):
      original_classes = get_diacritics_classes(original_word, arabic_letters, diacritic_classes)
      target_classes = get_diacritics_classes(target_word, arabic_letters, diacritic_classes)

      assert(len(original_classes) == len(target_classes))

      if len(original_classes) == 0:
        continue

      equal_classes = 0
      for (original_class, target_class) in zip(original_classes, target_classes):
        equal_classes += (original_class == target_class)

      if(equal_classes != len(original_classes)):
        word = remove_diacritics(original_word, diacritic_classes)
        try:
          freq[word] += 1
        except:
          freq[word] = 1
  return freq

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Count each words error frequency')
  parser.add_argument('-ofp', '--original-file-path', help='File path to original text', required=True)
  parser.add_argument('-tfp', '--target-file-path', help='File path to target text', required=True)
  parser.add_argument('-lim', '--freq-limit', help='Lowest frequency to show in the plot', required=True)
  args = parser.parse_args()

  with open(CONSTANTS_PATH + '/ARABIC_LETTERS_LIST.pickle', 'rb') as file:
    ARABIC_LETTERS_LIST = pkl.load(file)

  with open(CONSTANTS_PATH + '/CLASSES_LIST.pickle', 'rb') as file:
    CLASSES_LIST = pkl.load(file)

  limit = int(args.freq_limit)
  
  freq = count_error_frequency(args.original_file_path, args.target_file_path, ARABIC_LETTERS_LIST, CLASSES_LIST)
  freq = {get_display(arabic_reshaper.reshape(key)) : value for key, value in freq.items() if value > limit}
  freq = list(zip(*sorted(freq.items(), key=lambda kv: kv[1])))

  figure(figsize=(15, 6), dpi=200, edgecolor='k')
  plt.bar(freq[0], freq[1], align='center')
  plt.title('Word Error Frequencies')
  plt.ylabel('Frequency')
  plt.xlabel('Words')
  plt.xticks(rotation='vertical')
  plt.savefig('Word Error Frequencies')
