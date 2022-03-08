from http.client import _DataType
import zipfile
import numpy as np
import matplotlib.pyplot as plt
import cv2
import lmdb
import caffe

N = 3065*0.6

X = np.zeros((N, 3, 227, 227), dtype=np.uint8)
y = np.zeros(N, dtype=np.int64)

#set the datamap 10times greater that what we actually need 

map_size = X.nbytes * 10
env = lmdb.open('Dataset/train_lmdb', map_size=map_size)

zip = zipfile.ZipFile('Dataset/raw_data.zip','r')
filename = None
for filename in range (1,N):
    f = zip.read('{filename}.mat')

    img = f['cjdata']['image']
    img = np .array(img, dtype=np.float64)
    
    label = f['cjdata']['label'][0][0]

    X[filename-1] = img
    y[filename-1] = label
print ("--- data loaded ---")

print("--- converting data to lmdb ---")
with env.begin(write=True) as txn:
    # txn is a Transaction object
    for i in range(N):
        datum = caffe.proto.caffe_pb2.Datum()
        datum.channels = X.shape[1]
        datum.height = X.shape[2]
        datum.width = X.shape[3]
        datum.data = X[i].tobytes()  # or .tostring() if numpy < 1.9
        datum.label = int(y[i])
        str_id = '{:08}'.format(i)

        # The encode is only essential in Python 3
        txn.put(str_id.encode('ascii'), datum.SerializeToString())
print ("--- Done ---")
print("training dataset size: {N}")