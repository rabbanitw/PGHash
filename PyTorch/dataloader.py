from itertools import islice
import numpy as np


def data_generator_test(files, batch_size):
    while 1:
        lines = []
        for file in files:
            with open(file, 'r', encoding='utf-8') as f:
                header = f.readline()  # ignore the header

                while True:
                    temp = len(lines)
                    lines += list(islice(f, batch_size-temp))
                    if len(lines)!=batch_size:
                        break
                    inds = [[], []]
                    vals = []
                    y_batch = [None for i in range(len(lines))]
                    count = 0
                    for line in lines:
                        itms = line.strip().split(' ')
                        y_batch[count] = [int(itm) for itm in itms[0].split(',')]
                        inds[0] += [count]*len(itms[1:])
                        inds[1] += [int(itm.split(':')[0]) for itm in itms[1:]]
                        vals += [float(itm.split(':')[1]) for itm in itms[1:]]
                        count += 1
                    lines = []
                    yield (inds, vals, y_batch)


def data_generator_train(files, batch_size, n_classes):
    while 1:
        lines = []
        for file in files:
            with open(file, 'r', encoding='utf-8') as f:
                header = f.readline()  # ignore the header
                while True:
                    temp = len(lines)
                    lines += list(islice(f,batch_size-temp))
                    if len(lines)!=batch_size:
                        break
                    inds = [[], []]
                    vals = []
                    y_batch = np.zeros([batch_size,n_classes], dtype=float)
                    count = 0
                    for line in lines:
                        itms = line.strip().split(' ')
                        y_inds = [int(itm) for itm in itms[0].split(',')]
                        for i in range(len(y_inds)):
                            y_batch[count, y_inds[i]] = 1.0/len(y_inds)
                            # y_batch[count,y_inds[i]] = 1.0
                        inds[0] += [count]*len(itms[1:])
                        inds[1] += [int(itm.split(':')[0]) for itm in itms[1:]]
                        vals += [float(itm.split(':')[1]) for itm in itms[1:]]
                        count += 1
                    lines = []
                    yield (inds, vals, y_batch)
