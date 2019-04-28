import argparse
import matplotlib.pyplot as plt

def restore_model_accuracy_and_loss(file_path):
  loss = list()
  acc = list()
  val_loss = list()
  val_acc = list()

  with open(file_path, 'r') as file:
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
  parser = argparse.ArgumentParser(description='Restores the accuracy and loss figure from training logs')
  parser.add_argument('-in', '--file-path', required=True)
  args = parser.parse_args()

  restore_model_accuracy_and_loss(args.file_path)
