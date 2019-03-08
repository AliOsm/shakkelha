import pickle as pkl

# Constants
CONSTANTS_PATH = '../../helpers/constants'
DATASET_PATH = '../../dataset'

with open(CONSTANTS_PATH + '/DIACRITICS_LIST.pickle', 'rb') as file:
  DIACRITICS_LIST = pkl.load(file)

def remove_diacritics(content):
  return content.translate(str.maketrans('', '', ''.join(DIACRITICS_LIST)))

with open(DATASET_PATH + '/test.txt', 'r') as file:
  test_data = file.readlines()

new_test_data = list()
for line in test_data:
  line = line.strip()
  tmp = ''
  length = 0
  for word in line.split():
    if length >= 110:
      new_test_data.append(tmp)
      tmp = ''
      length = 0
    if tmp != '':
      tmp += ' '
      length += 1
    tmp += word
    length += len(remove_diacritics(word))
  if tmp != '':
    new_test_data.append(tmp)

with open('split_test.txt', 'w') as file:
  file.write('\n'.join(new_test_data))
