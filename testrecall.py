import logging
logging.basicConfig(level=logging.NOTSET)

import numpy 
import sys 
import nmslib 
import time 
import math 
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
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


# all_data_matrix = numpy.loadtxt('/home/ubuntu/Lulingling/testnmslib/final128_10K.txt')
# query_matrix = all_data_matrix[0:1000]
# data_matrix = all_data_matrix[1000:10000]


#all_data_matrix = ivecs_read("/home/ubuntu/Lulingling/testnmslib/sift/sift_groundtruth.ivecs") 

all_data_matrix = fvecs_read("/home/ubuntu/Lulingling/testnmslib/sift/sift_base.fvecs")
all_data_matrix_query = fvecs_read("/home/ubuntu/Lulingling/testnmslib/sift/sift_query.fvecs")  

# query_matrix = all_data_matrix[0:1000]
# data_matrix = all_data_matrix[1000:100000]
query_matrix = all_data_matrix_query[0:1000]
data_matrix = all_data_matrix[0:1000000]#800100   1000000 



#print('Index-time parameters', query_matrix)

num_threads = 1
# index_time_params = {'bucketSize': 10, 'chunkBucket': 1
#                     }
index_time_params = {'tuneK': 10, 'desiredRecall': 0.98,'bucketSize': 10, 'chunkBucket': 1
                    }
 

# Number of neighbors 
K=10
# Space name should correspond to the space name 
# used for brute-force search
space_name='l2'
# Intitialize the library, specify the space, the type of the vector and add data points 
index = nmslib.init(method='vptree', space=space_name, data_type=nmslib.DataType.DENSE_VECTOR) 

query_time_params = {'alphaLeft': 2.0,'alphaRight': 2.0,'expLeft': 1, 'expRight': 1,'maxLeavesToVisit': 1}
#print('Setting query-time parameters', query_time_params)
#index.setQueryTimeParams(query_time_params)


index.addDataPointBatch(data_matrix) 

# Create an index
start = time.time()
index.createIndex(index_time_params) 
end = time.time() 
print('Index-time parameters', index_time_params)
print('Indexing time = %f' % (end-start))


# Querying
query_qty = query_matrix.shape[0]
start = time.time() 
nbrs = index.knnQueryBatch(query_matrix, k = K, num_threads = num_threads)
end = time.time() 
print('kNN time total=%f (sec), per query=%f (sec), per query adjusted for thread number=%f (sec)' % 
      (end-start, float(end-start)/query_qty, num_threads*float(end-start)/query_qty)) 

#index.saveIndex('dense_index_vptree.bin')


# Computing gold-standard data 
print('Computing gold-standard data')

start = time.time()
sindx = NearestNeighbors(n_neighbors=K, metric='l2', algorithm='brute').fit(data_matrix)
end = time.time()

print('Brute-force preparation time %f' % (end - start))

start = time.time() 
gs = sindx.kneighbors(query_matrix)
end = time.time()

print('brute-force kNN time total=%f (sec), per query=%f (sec)' % 
      (end-start, float(end-start)/query_qty) )


# Finally computing recall
recall=0.0
for i in range(0, query_qty):
  correct_set = set(gs[1][i])
  ret_set = set(nbrs[i][0])
  recall = recall + float(len(correct_set.intersection(ret_set))) / len(correct_set)
recall = recall / query_qty
print('kNN recall %f' % recall)


