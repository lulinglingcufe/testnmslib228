import numpy as np


# all_data_matrix = np.random.randn(100, 12).astype(np.float32)

# query_matrix = all_data_matrix[0:2]
# print(query_matrix) 


# query_matrix = all_data_matrix[0:600]

def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')#int32
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()


def fvecs_read(fname):
    return ivecs_read(fname).view('float32')


d = fvecs_read("/home/ubuntu/lulingling/testnmslib/sift/sift_base.fvecs")  #sift_query.fvecs  sift_groundtruth.ivecs  sift_groundtruth.ivecs
print(d[0:4])  
