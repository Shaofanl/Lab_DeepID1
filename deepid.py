import os
import caffe
from caffe import layers as L
from caffe import params as P
import numpy as np
import gc

from config import *

def getMulticlassNet(lmdb, batch_size):
    # LeNet for lfw
    n = caffe.NetSpec()

    n.data, n.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=lmdb,
        transform_param=dict(scale=1./255), ntop=2)

    n.conv1 = L.Convolution(n.data, kernel_size=4, stride=1, num_output=20, weight_filler=dict(type='xavier'))
    n.pool1 = L.Pooling(n.conv1, kernel_size=2, stride=2, pool=P.Pooling.MAX)

    n.conv2 = L.Convolution(n.pool1, kernel_size=3, stride=1, num_output=40, weight_filler=dict(type='xavier'))
    n.pool2 = L.Pooling(n.conv2, kernel_size=2, stride=2, pool=P.Pooling.MAX)

    n.conv3 = L.Convolution(n.pool2, kernel_size=3, stride=1, num_output=60, weight_filler=dict(type='xavier'))
    n.pool3 = L.Pooling(n.conv3, kernel_size=2, stride=2, pool=P.Pooling.MAX)

    n.conv4 = L.Convolution(n.pool3, kernel_size=2, stride=1, num_output=80, weight_filler=dict(type='xavier'))

    n.conv4deepid = L.InnerProduct(n.conv4, num_output=160, weight_filler=dict(type='xavier'))
    n.pool3deepid = L.InnerProduct(n.pool3, num_output=160, weight_filler=dict(type='xavier'))

    n.deepid__ = L.Eltwise(n.conv4deepid, n.pool3deepid, eltwise_param={'operation': 1})
    n.deepid = L.ReLU(n.deepid__)

    n.fc = L.InnerProduct(n.deepid, num_output=4000, weight_filler=dict(type='xavier'))
    n.loss = L.SoftmaxWithLoss(n.fc, n.label)

    return n.to_proto()

def createSolver(filename, **kwargs):
    with open(filename, 'w') as f:
        for key, val in kwargs.iteritems():
            f.write('{}: {}\n'.format(key, val))
    return filename

def runSolver(solver, predict_layer, test_set=False, rounds=10, epochs=100):
    print [(k, v.data.shape) for k, v in solver.net.blobs.items()]
    print [(k, v[0].data.shape) for k, v in solver.net.params.items()]

    # losses will also be stored in the log

    # the main solver loop
    for it in range(rounds):
        # accuracy
        correct = 0
        for i in xrange(10):
            solver.net.forward()
            correct += (solver.net.blobs[predict_layer].data.argmax(1) == solver.net.blobs['label'].data).sum()
        print 'Train: [%04d]: %04d/1000=%.5f%%' % (it, correct, correct/1e3*1e2)

        solver.step(epochs)  # SGD by Caffe

if __name__ == '__main__':
    caffe.set_mode_gpu()
    caffe.set_device(1)

    with open('./lfw_multiclass.prototxt', 'w') as f:
        f.write(str(getMulticlassNet(MULTICLASS_LMDB, 100)))
    multiclass_solver = caffe.SGDSolver(
        createSolver('multiclassSolver.prototxt', 
            train_net='"./lfw_multiclass.prototxt"',
            base_lr=5e-2,
#            rms_decay=0.02,
#            momentum=0.9,
            weight_decay=0.001,
            lr_policy='"inv"',
            gamma=0.001,
            power=0.75,
            display=10,
            solver_mode="GPU",
            snapshot=100,
            snapshot_prefix='"/home/share/shaofan/lfw_caffe/snapshot/"',
            ))
    runSolver(multiclass_solver, 'fc', rounds=100, epochs=100)


