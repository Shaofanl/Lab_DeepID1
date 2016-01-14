##
#    IPYTHON
#   net = caffe.Net(caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt',
#           caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel',
#           caffe.TEST)
# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
#   transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
#   transformer.set_transpose('data', (2,0,1))
#   transformer.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1)) # mean pixel
#   transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
#   transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB
# set net to batch size of 50
#   net.blobs['data'].reshape(50,3,227,227)
#   net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(caffe_root + 'examples/images/cat.jpg'))
#ut = net.forward()
#   print("Predicted class is #{}.".format(out['prob'][0].argmax()))

import caffe
import os
import sys
from config import *
import numpy as np

def get_2partdata(filename, transformer):
    x = []
    y = []
    with open(filename, 'r') as f:
        n = int(f.readline().strip())
        # the same one
        for label in range(n):
            print '\rList: %d/%d'%(label, 2*n),
            name, num1, num2 = f.readline().strip().split('\t')
            img1 = transformer.preprocess('data', caffe.io.load_image(os.path.join(DATA_ROOT, name, '%s_%04d.jpg'%(name, int(num1)))))
            img2 = transformer.preprocess('data', caffe.io.load_image(os.path.join(DATA_ROOT, name, '%s_%04d.jpg'%(name, int(num2)))))
            x.extend([img1, img2])
            y.append(1)
            sys.stdout.flush()
        # the diff face
        for label in range(n):
            print '\rList: %d/%d'%(n+label, 2*n),
            name1, num1, name2, num2 = f.readline().strip().split('\t')
            img1 = transformer.preprocess('data', caffe.io.load_image(os.path.join(DATA_ROOT, name1, '%s_%04d.jpg'%(name1, int(num1)))))
            img2 = transformer.preprocess('data', caffe.io.load_image(os.path.join(DATA_ROOT, name2, '%s_%04d.jpg'%(name2, int(num2)))))
            x.extend([img1, img2])
            y.append(0)
            sys.stdout.flush()
    return np.array(x), np.array(y)

def get_feature(net, x):
    print '[get_feature({})]: shape={}'.format(id(x), x.shape)
    # deepid
    net.blobs['data'].reshape(*((x.shape[0], ) + net.blobs['data'].data.shape[1:]))
    net.blobs['data'].data[...] = x
    net.forward()
    print '[get_feature({})]: feature_shape={}'.format(id(x), net.blobs['deepid'].data.shape)
    return net.blobs['deepid'].data

class Threshold(object):
    def __init__(self):
        pass

    def fit(self, x, y, tol=1e-5):
        sortedx = sorted(x)
        l = 0.0
        r = 1.0
        while l+tol < r:
            lmid = (2*l+r)/3.0
            rmid = (l+2*r)/3.0
            if (self.predict(x, lmid)==y).sum() <= (self.predict(x, rmid)==y).sum():
                l = lmid
            else:
                r = rmid
        self.p = (l+r)/2.0 

    def predict(self, x, p=None):
        if p == None: p = self.p
        ind = int(x.shape[0]*p)
        if ind == x.shape[0]: ind -= 1
        threshold = sorted(x)[ind]
        y = np.zeros((x.shape[0],))
        y[x >= threshold] = 1
        return y
        

def test(clf, trainx, trainy, testx, testy, **kwargs):
    clf.fit(trainx, trainy)
    acc = (clf.predict(trainx, **kwargs) == trainy).sum()
    print 'Train: {}/{}'.format(acc, trainy.shape[0])

    acc = (clf.predict(testx, **kwargs) == testy).sum()
    print 'Test: {}/{}'.format(acc, testy.shape[0])


if __name__ == '__main__':
    net = caffe.Net('./lfw_multiclass.deploy.prototxt', 
                    '/home/share/shaofan/lfw_caffe/snapshot/_iter_2100.caffemodel', 
                    caffe.TEST)
# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))
#    transformer.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1)) # mean pixel
    transformer.set_raw_scale('data', 1.0)  # the reference model operates on images in [0,255] range instead of [0,1]
    transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB
# set net to batch size of 100
#    net.blobs['data'].reshape(*net.blobs['data'].data.shape[1:])
#    img = transformer.preprocess('data', caffe.io.load_image('/home/share/lfw/Zhang_Ziyi/Zhang_Ziyi_0001.jpg'))
#    net.blobs['data'].data[0] = img
#    net.blobs['data'].data[1] = img
# rock and roll
#    net.forward()
#   print("Predicted class is #{}.".format(out['prob'][0].argmax()))


# ===================================testing=============================================
    trainx, trainy = get_2partdata(os.path.join(LFW_ROOT, 'pairsDevTrain.txt'), transformer)
    testx, testy= get_2partdata(os.path.join(LFW_ROOT, 'pairsDevTest.txt'), transformer)
    
    print 'fetching features of trainx'
    trainx = get_feature(net, trainx).reshape(trainy.shape[0], 2, -1) 
    testx = get_feature(net, testx).reshape(testy.shape[0], 2, -1) 
    print 'feature shape:', trainx.shape
    print trainx.shape
    print testx.shape

    methods = ['diff', 'cosdiff', 'svm']
    
    from sklearn import svm
    from sklearn.preprocessing import normalize
    # svm
    if 'svm' in methods:
        print 'Concatenate + SVM'
        __trainx = np.concatenate([trainx[:, 0, :], trainx[:, 1, :]], 1)
        __testx = np.concatenate([testx[:, 0, :], testx[:, 1, :]], 1)
        clf = svm.SVC(kernel='linear', probability=True, verbose=False, max_iter=100000)
        test(clf, __trainx, trainy, __testx, testy)

    if 'svm' in methods:
        print 'Minues + SVM'
        __trainx = normalize(trainx[:, 0, :] - trainx[:, 1, :])
        __testx = testx[:, 0, :] - testx[:, 1, :]
        clf = svm.SVC(kernel='linear', probability=True, verbose=False, max_iter=100000)
        test(clf, __trainx, trainy, __testx, testy)

        print 'Minues + SVM(rbf)'
        clf = svm.SVC(kernel='rbf', probability=True, verbose=False, max_iter=100000)
        test(clf, __trainx, trainy, __testx, testy)

    # diff threshold
    if 'diff' in methods:
        print 'Minues + L2 norm + Threshold: '
        __trainx = ((trainx[:, 0, :] - trainx[:, 1, :])**2).sum(1)
        __testx = ((testx[:, 0, :] - testx[:, 1, :])**2).sum(1)

        clf = Threshold()
        test(clf, __trainx, trainy, __testx, testy)

    # cosdiff threshold
    if 'cosdiff' in methods:
        print 'Cos difference + Threshold: '
        __trainx = (trainx[:, 0, :] * trainx[:, 1, :]).sum(1)/np.sqrt((trainx[:, 0, :]**2).sum(1))/np.sqrt((trainx[:, 1, :]**2).sum(1))
        __testx = (testx[:, 0, :] * testx[:, 1, :]).sum(1)/np.sqrt((testx[:, 0, :]**2).sum(1))/np.sqrt((testx[:, 1, :]**2).sum(1))

        clf = Threshold()
        test(clf, __trainx, trainy, __testx, testy)



