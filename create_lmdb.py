import os
from shutil import rmtree
import Image as I
import numpy as np
from numpy import random
import lmdb
import caffe

from config import *

def getFromList(filename):
    x = []
    y = []
    with open(filename, 'r') as f:
        n = int(f.readline().strip())
#        n = DEBUG_N
        for label in range(n):
            print '\rList: %d/%d'%(label, n),
            name, num = f.readline().strip().split('\t')
            for ind in range(int(num)):
                img = np.array( I.open(os.path.join(DATA_ROOT, name, '%s_%04d.jpg'%(name, ind+1))).resize(SIZE) ).swapaxes(0, 2)
                x.append(img)
                y.append(label)
    return x, y

def putToLMDB(x, y, lmdbname):
    zipped = zip(x, y)
    random.shuffle(zipped)
    x, y = zip(*zipped)

    N = len(x)
    env = lmdb.open(lmdbname, map_size=N*x[0].nbytes*10)
    info = x[0].shape
    print info
    with env.begin(write=True) as txn:
        for i in range(N):
            print '\rLMDB: %d/%d'%(i, N),
            datum = caffe.proto.caffe_pb2.Datum()
            datum.channels, datum.height, datum.width = info
            datum.data = x[i].tostring() # numpy >= 1.9: tobytes()
            datum.label = y[i]
            str_id = '{:08}'.format(i)

            txn.put(str_id.encode('ascii'), datum.SerializeToString())

def create_multiclass_lmdb():
    x, y = getFromList(os.path.join(LFW_ROOT, 'peopleDevTrain.txt'))
    putToLMDB(x, y, os.path.join(LMDB_ROOT, 'multiclass'))

def create_binclass_lmdb():
    def getPairs(x, y, N=8000):
        pairs_x = [] 
        pairs_y = [] 

        ycount = len(set(y))
        y2ind = [[] for i in range(ycount)]
        for ind, ele in enumerate(y):
            y2ind[ele].append(ind)

        each_N = N/2
        # matched pair
        for i in range(each_N):
            print '\rPair: %d/%d'%(i, N),
            while True:
                y = random.randint(0, ycount)
                if len(y2ind[y]) > 1: break
            x1, x2 = random.choice(y2ind[y], 2, False)
            
            pairs_x.append(np.concatenate([x[x1], x[x2]]))
            pairs_y.append(1)

        #dismatched pair
        for i in range(each_N):
            print '\rPair: %d/%d'%(i+each_N, N),
            y1, y2 = random.choice(range(ycount), 2, False)
            x1 = random.choice(y2ind[y1])
            x2 = random.choice(y2ind[y2])

            pairs_x.append(np.concatenate([x[x1], x[x2]]))
            pairs_y.append(1)
        return pairs_x, pairs_y

    x, y = getFromList(os.path.join(LFW_ROOT, 'peopleDevTrain.txt'))
    train_x, train_y = getPairs(x, y, 1000)
    putToLMDB(train_x, train_y, os.path.join(LMDB_ROOT, 'bin_train'))

    x, y = getFromList(os.path.join(LFW_ROOT, 'peopleDevTest.txt'))
    test_x, test_y = getPairs(x, y, 300)
    putToLMDB(test_x, test_y, os.path.join(LMDB_ROOT, 'bin_test'))
    

if __name__ == '__main__':
    print('Destroying previous lmdb ...')
    try:
        rmtree(LMDB_ROOT)
        print('\tDone.')
    except:
        print('\tFailed.')
        pass

    print('Creating new lmdb ...')
    os.mkdir(LMDB_ROOT)
    print('\tDone.')

    print('Creating multiclass database ...')
    create_multiclass_lmdb()
    print('\tDone.')

#   print('Creating binclass database ...')
#   create_binclass_lmdb()
#   print('\tDone.')


