import zipfile
import numpy as np
import matplotlib.pyplot as plt
import h5py
import cv2
import lmdb
import caffe

def convert(path,start, N):
    X = np.zeros((N, 3, 224, 224), dtype=np.uint8)
    y = np.zeros(N, dtype=np.int64)

    #set the datamap 10times greater that what we actually need 

    map_size = X.nbytes * 10

    filename = None
    for filename in range (start,N+start):
        with h5py.File('/home/ouma/Desktop/Caffe_models/ResNet50_BT/Dataset/raw_data/{}/{}.mat'.format(path,filename), 'r') as f:

            img = f['cjdata']['image']
            img = np .array(img, dtype=np.float32)
            img = cv2.normalize(img, None,alpha =0,beta= 225,norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
            img = img.astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            img = cv2.resize(img, (224, 224))
            img =np.reshape(img, (3,224,224))
            #plt.imshow(np.reshape(img,(224,224,3)))
            #plt.show()
            X[filename-start] = img

            label = f['cjdata']['label'][0][0]
            y[filename-start] = label.astype(np.int64)
            
    print ("--- data loaded ---")

    print("--- converting data to lmdb ---")
    env = lmdb.open('ResNet50_BT/Dataset/{}_lmdb'.format(path), map_size=map_size)
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
    print("{} dataset size: {}".format(path, N))

convert("train_1-1532",1,1532)
convert("test_2299-3064",2299, 3064-2299)