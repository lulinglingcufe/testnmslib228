import logging
logging.basicConfig(level=logging.NOTSET)

import numpy 
import sys 
import nmslib 
import time 
import math 
# from sklearn.neighbors import NearestNeighbors
# from sklearn.model_selection import train_test_split
print(sys.version)
print("NMSLIB version:", nmslib.__version__)

# Just read the data
#all_data_matrix = numpy.loadtxt('/home/ubuntu/lulingling/testnmslib/final128_10K.txt')
# all_data_matrix = numpy.random.randn(1000, 128).astype(numpy.float32)


def ivecs_read(fname):
    a = numpy.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()


def fvecs_read(fname):
    return ivecs_read(fname).view('float32')


all_data_matrix = fvecs_read("/home/ubuntu/lulingling/testnmslib/sift/sift_base.fvecs") 


query_matrix = all_data_matrix[0:8]
data_matrix = all_data_matrix[1000:2000]












# query_matrix = all_data_matrix[0:600]
# data_matrix = all_data_matrix[600:10000]
# Create a held-out query data set
# (data_matrix, query_matrix) = train_test_split(all_data_matrix, test_size = 0.1)

# print("# of queries %d, # of data points %d"  % (query_matrix.shape[0], data_matrix.shape[0]) )

# Set index parameters
# These are the most important onese
M = 15
efC = 100

num_threads = 1
index_time_params = {'M': M, 'indexThreadQty': num_threads, 'efConstruction': efC, 'post' : 0}
print('Index-time parameters', index_time_params)

# Number of neighbors 
#K=10
K=5 

# Space name should correspond to the space name 
# used for brute-force search
space_name='l2'# l2  cosinesimil
# Intitialize the library, specify the space, the type of the vector and add data points 
index = nmslib.init(method='hnsw', space=space_name, data_type=nmslib.DataType.DENSE_VECTOR) 
index.addDataPointBatch(data_matrix) 



# Create an index
start = time.time()
index.createIndex(index_time_params) 
end = time.time() 
print('Index-time parameters', index_time_params)
print('Indexing time = %f' % (end-start))

# Setting query-time parameters
efS = 100
query_time_params = {'efSearch': efS}
print('Setting query-time parameters', query_time_params)
index.setQueryTimeParams(query_time_params)


# Querying
query_qty = query_matrix.shape[0]
start = time.time() 
nbrs = index.knnQueryBatch(query_matrix, k = K, num_threads = num_threads)
end = time.time() 
print('kNN time total=%f (sec), per query=%f (sec), per query adjusted for thread number=%f (sec)' % 
      (end-start, float(end-start)/query_qty, num_threads*float(end-start)/query_qty)) 

print("query_qty\n",query_qty)

print("query_matrix\n",query_matrix)

print("query_matrix[0]\n",query_matrix[0])

print("query_matrix.shape\n",query_matrix.shape)

#print("nbrs.shape",nbrs.shape)

print("nbrs",nbrs[0])

for i in range(0, 8):
  ret_set = set(nbrs[i][0])
  print("ret_set",ret_set)



all_groundtruth_matrix = ivecs_read("/home/ubuntu/lulingling/testnmslib/sift/sift_groundtruth.ivecs") 


groundtruth_matrix = all_groundtruth_matrix[0:8]

print("groundtruth_matrix\n",groundtruth_matrix)


