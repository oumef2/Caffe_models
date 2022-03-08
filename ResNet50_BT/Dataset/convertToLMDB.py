import zipfile
import numpy as np
import matplotlib.pyplot as plt
import h5py
import cv2
import lmdb
import caffe

N = int(3065*0.6)
N = 766
X = np.zeros((N, 3, 224, 224), dtype=np.uint8)
y = np.zeros(N, dtype=np.int64)

#set the datamap 10times greater that what we actually need 

map_size = X.nbytes * 10



filename = None
for filename in range (1,N+1):
    with h5py.File('/home/ouma/Desktop/Caffe_models/ResNet50_BT/Dataset/raw_data/brainTumorDataPublic_1-766/{}.mat'.format(filename), 'r') as f:

        img = f['cjdata']['image']
        img = np .array(img, dtype=np.float32)
        img = cv2.normalize(img, None,alpha =0,beta= 225,norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
        img = img.astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = cv2.resize(img, (224, 224))
        img =np.reshape(img, (3,224,224))
        #plt.imshow(np.reshape(img,(224,224,3)))
        #plt.show()
        X[filename-1] = img

        label = f['cjdata']['label'][0][0]
        y[filename-1] = label.astype(np.int64)
        
print ("--- data loaded ---")

print("--- converting data to lmdb ---")
env = lmdb.open('train_lmdb', map_size=map_size)
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
