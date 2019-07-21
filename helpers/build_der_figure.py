import argparse
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
from os import sep, listdir
from os.path import isdir, join

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Builds DER Figure')
  parser.add_argument('-in', '--folder-path', required=True)
  args = parser.parse_args()

  folders = [join(args.folder_path, folder) for folder in listdir(args.folder_path) if isdir(join(args.folder_path, folder))]
  folders = list(sorted(folders, key=lambda item: int(item.split(sep)[-1].split('_')[0])))
  
  colors = ['r', 'b', 'k']

  fig, ax = plt.subplots(num=None, figsize=(10, 5), dpi=100, facecolor='w', edgecolor='k')

  for color, folder in zip(colors, folders):
    print(folder)
    small_logs_file = join(folder, 'small_data', 'info', 'logs.txt')
    big_logs_file = join(folder, 'big_data', 'info', 'logs.txt')
    
    with open(small_logs_file, 'r') as file:
      small_logs_file = file.readlines()
    with open(big_logs_file, 'r') as file:
      big_logs_file = file.readlines()

    small_der = list()
    for line in small_logs_file:
      if 'Validation' in line:
        small_der.append(float(line.strip().split()[-1]))
    big_der = list()
    for line in big_logs_file:
      if 'Validation' in line:
        big_der.append(float(line.strip().split()[-1]))

    plt.plot([i for i in range(5, 51, 5)], small_der, '%s%s--' % (color, 's'))
    plt.plot([i for i in range(5, 51, 5)], big_der, '%s%s-' % (color, 'o'))

  plt.ylabel('DER', fontsize=20)
  plt.xlabel('Epoch', fontsize=20)

  plt.xticks([i for i in range(5, 51, 5)], fontsize=16)

  for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(16)

  handles = list()
  for color, folder in zip(colors, folders):
    model_name = folder.split(sep)[-1].split('_')[1].title()
    if model_name == 'Crf':
      model_name = model_name.upper()
    if model_name == 'Normalized':
      model_name = 'BNG'
    handles.append(Line2D([0], [0], color=color, lw=4, label=model_name + ' Model'))
  handles.append(Line2D([0], [0], marker='s', label='w/o extra train'))
  handles.append(Line2D([0], [0], marker='o', label='w/ extra train'))

  plt.legend(handles=handles, loc='best', fontsize=20)

  plt.tight_layout()

  plt.show()
