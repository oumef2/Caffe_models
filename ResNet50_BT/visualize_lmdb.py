import numpy as np
import lmdb
import caffe
import matplotlib.pyplot as plt

env = lmdb.open('ResNet50_BT/Dataset/train_1-2000_lmdb', readonly=True)
with env.begin() as txn:
    raw_datum = txn.get(b'00000000')

datum = caffe.proto.caffe_pb2.Datum()
datum.ParseFromString(raw_datum)

flat_x = np.frombuffer(datum.data, dtype=np.uint8)
x = flat_x.reshape(datum.channels, datum.height, datum.width)
y = datum.label
print(x.shape)
print(y)
plt.imshow(np.transpose(x,(2, 1, 0)))
plt.show()
with env.begin() as txn:
    cursor = txn.cursor()
    for key, value in cursor:
        print(key, value)