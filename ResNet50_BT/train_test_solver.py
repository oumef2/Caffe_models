import caffe
import numpy as np
import matplotlib.pyplot as plt

from pylab import *

#caffe.set_device(0)
caffe.set_mode_cpu()

# recent modification, remove if it doesn't work

print("Initialized caffe")

solver = caffe.SGDSolver('ResNet50_BT/solver.prototxt')

#[print((k, v[0].data.shape)) for k, v in solver.net.params.items()]
[print((k, v.data.shape)) for k, v in solver.net.blobs.items()]

solver.solve()

# we use a little trick to tile the first eight images
imshow(solver.net.blobs['data'].data[:8, 0].transpose(2, 1, 0).reshape(224, 8*224), cmap='gray'); axis('off')
print ('train labels:', solver.net.blobs['label'].data[:8])