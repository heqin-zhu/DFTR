import os
import sys
import shutil
import random


def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def main(pre='.'):
    train_num_dirs = {'NJU2K': 1500, 'NLPR': 700}
    types = ['RGB', 'depth', 'GT']

    for dataname, num in train_num_dirs.items():
        d  = os.path.join(pre, dataname)
        names = [i[:-4] for i in os.listdir(os.path.join(d, 'depth'))]
        names.sort()
        random.seed(42) # important
        random.shuffle(names) # 
        train_path = d + '_train' # TODO
        test_path = d + '_test'
        mkdir(train_path)
        mkdir(test_path)
        for tp in types:
            prefix = os.path.join(d, tp)
            mkdir(os.path.join(train_path, tp))
            mkdir(os.path.join(test_path, tp))
            filetype = '.jpg' if tp == 'RGB' else '.png'

            for i in range(len(names)):
                filename = names[i] + filetype
                src = os.path.join(prefix, filename)
                if i < num:
                    shutil.copy(src, os.path.join(train_path, tp, filename))
                else:
                    shutil.copy(src, os.path.join(test_path, tp, filename))


if __name__ == '__main__':
    dirname, filename = os.path.split(os.path.abspath(sys.argv[0]))
    if len(sys.argv)>1:
        dirname = sys.argv[1]
    main(dirname)
