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

  fig = plt.figure(num=None, figsize=(8, 10), dpi=100, facecolor='w', edgecolor='k')

  ax = fig.add_subplot(2, 1, 1)
  ax.plot(acc, 'r--')
  ax.plot(val_acc, 'b-')
  ax.set_ylabel('Accuracy', fontsize=20)
  ax.set_xlabel('Epoch', fontsize=20)
  for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(16)
  for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(16)
  ax.legend(['Training', 'Validation'], loc='best', fontsize=16)

  ax = fig.add_subplot(2, 1, 2)
  ax.plot(loss, 'r--')
  ax.plot(val_loss, 'b-')
  ax.set_ylabel('Loss', fontsize=20)
  ax.set_xlabel('Epoch', fontsize=20)
  for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(16)
  for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(16)
  ax.legend(['Training', 'Validation'], loc='best', fontsize=16)

  plt.tight_layout()
  
  plt.show()

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Restores the accuracy and loss figure from training logs')
  parser.add_argument('-in', '--file-path', required=True)
  args = parser.parse_args()

  restore_model_accuracy_and_loss(args.file_path)
