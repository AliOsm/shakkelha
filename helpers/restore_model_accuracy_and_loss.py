import sys
import matplotlib.pyplot as plt

def restore_model_accuracy_and_loss(FILE_PATH):
  loss = list()
  acc = list()
  val_loss = list()
  val_acc = list()

  with open(FILE_PATH, 'r') as file:
    lines = file.readlines()

  for idx in range(1, len(lines), 2):
    target = lines[idx].split()
    loss.append(float(target[7]))
    acc.append(float(target[10]))
    val_loss.append(float(target[13]))
    val_acc.append(float(target[16]))

  plt.figure(num=None, figsize=(16, 6), dpi=100, facecolor='w', edgecolor='k')

  plt.subplot(1, 2, 1)
  plt.plot(acc, 'r--')
  plt.plot(val_acc, 'b-')
  plt.title('Model Accuracy')
  plt.ylabel('Accuracy')
  plt.xlabel('Epoch')
  plt.legend(['Training', 'Validation'], loc='best')

  plt.subplot(1, 2, 2)
  plt.plot(loss, 'r--')
  plt.plot(val_loss, 'b-')
  plt.title('Model Loss')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Training', 'Validation'], loc='best')

  plt.show()

if __name__ == '__main__':
  if len(sys.argv) != 2:
    sys.exit('usage: python %s [TRAINING_LOGS_FILE]' % sys.argv[0])

  FILE_PATH = sys.argv[1]

  restore_model_accuracy_and_loss(FILE_PATH)
