import logging
logging.basicConfig(level=logging.NOTSET)

import numpy 
import sys 
import nmslib 
import time 
import math 
import random

# from sklearn.neighbors import NearestNeighbors
# from sklearn.model_selection import train_test_split
print(sys.version)
print("NMSLIB version:", nmslib.__version__)


# Just read the data
# all_data_matrix = numpy.loadtxt('/home/ubuntu/lulingling/testnmslib/final128_10K.txt')
# all_data_matrix = numpy.random.randn(1000000, 128).astype(numpy.float32)

def ivecs_read(fname):
    a = numpy.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()


def fvecs_read(fname):
    return ivecs_read(fname).view('float32')


all_data_matrix = fvecs_read("/home/ubuntu/lulingling/testnmslib/sift/sift_base.fvecs")  # sift_query    .fvecs  sift_groundtruth.ivecs  sift_groundtruth.ivecs
# #print(d[0:4])  
all_data_matrix_query = fvecs_read("/home/ubuntu/lulingling/testnmslib/sift/sift_query.fvecs")  

#query_matrix = all_data_matrix[0:100]
#query_matrix = [all_data_matrix[70],all_data_matrix[70],all_data_matrix[70],all_data_matrix[70],all_data_matrix[70],all_data_matrix[70],all_data_matrix[70],all_data_matrix[70],all_data_matrix[70],all_data_matrix[70]]
#query_matrix = all_data_matrix[0:100]


#query_matrix = all_data_matrix_query[0:100]


query_matrix=[]
sample_list = random.sample(range(0,999),1) #random查询的数量
for sample_list_item in sample_list:
    query_matrix.append(all_data_matrix_query[sample_list_item])


data_matrix = all_data_matrix[0:100000]#800100   900000
print('Load data\n')



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
K=10
# Space name should correspond to the space name 
# used for brute-force search
space_name='l2'#
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


# Querying 查询?
#query_qty = query_matrix.shape[0]
start = time.time() 
nbrs = index.knnQueryBatch(query_matrix, k = K, num_threads = num_threads)
end = time.time() 
#print('kNN time total=%f (sec), per query=%f (sec), per query adjusted for thread number=%f (sec)' % 
#       (end-start, float(end-start)/query_qty, num_threads*float(end-start)/query_qty)) 











# # Computing gold-standard data 正确的knn数据 
# print('Computing gold-standard data')

# start = time.time()
# sindx = NearestNeighbors(n_neighbors=K, metric='l2', algorithm='brute').fit(data_matrix)
# end = time.time()

# print('Brute-force preparation time %f' % (end - start))

# start = time.time() 
# gs = sindx.kneighbors(query_matrix)
# end = time.time()

# print('brute-force kNN time total=%f (sec), per query=%f (sec)' % 
#       (end-start, float(end-start)/query_qty) )

# # Finally computing recall
# recall=0.0
# for i in range(0, query_qty):
#   correct_set = set(gs[1][i])
#   ret_set = set(nbrs[i][0])
#   recall = recall + float(len(correct_set.intersection(ret_set))) / len(correct_set)
# recall = recall / query_qty
# print('kNN recall %f' % recall)


# Save a meta index, but no data!
#index.saveIndex('dense_index_optim.bin', save_data=False)


#存储索引？
#index.saveIndex('dense_index_nonoptim.bin', save_data=True)

# # Re-intitialize the library, specify the space, the type of the vector.
# newIndex = nmslib.init(method='hnsw', space=space_name, data_type=nmslib.DataType.DENSE_VECTOR) 
# # For an optimized L2 index, there's no need to re-load data points, but this would be required for
# # non-optimized index or any other methods different from HNSW (other methods can save only meta indices)
# newIndex.addDataPointBatch(data_matrix) 

# # # Re-load the index and re-run queries
# #newIndex.loadIndex('dense_index_optim.bin')
# newIndex.loadIndex('dense_index_nonoptim.bin', load_data=True)

# # # Setting query-time parameters and querying
# print('Setting query-time parameters', query_time_params)
# newIndex.setQueryTimeParams(query_time_params)

# query_qty = query_matrix.shape[0]
# start = time.time() 
# new_nbrs = newIndex.knnQueryBatch(query_matrix, k = K, num_threads = num_threads)
# end = time.time() 
# print('kNN time total=%f (sec), per query=%f (sec), per query adjusted for thread number=%f (sec)' % 
#       (end-start, float(end-start)/query_qty, num_threads*float(end-start)/query_qty)) 


# # Finally computing recall for the new result set
# recall=0.0
# for i in range(0, query_qty):
#   correct_set = set(gs[1][i])
#   ret_set = set(new_nbrs[i][0])
#   recall = recall + float(len(correct_set.intersection(ret_set))) / len(correct_set)
# recall = recall / query_qty
# print('kNN recall %f' % recall)

