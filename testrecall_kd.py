import logging
logging.basicConfig(level=logging.NOTSET)

import numpy 
import sys 
import time 
import math 
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KDTree



print(sys.version)

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
query_matrix = all_data_matrix[0:1000]
data_matrix = all_data_matrix[1000:100000]
#data_matrix = all_data_matrix[1000:2000] 实验数据


#测试数据
# query_matrix = all_data_matrix[0:5]
# data_matrix = all_data_matrix[10:100]


query_qty = query_matrix.shape[0]
K=10
num_threads=1
start = time.time() 
#X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
kdt = KDTree(data_matrix, leaf_size=10, metric='euclidean')
#kdt.query(data_matrix, k=10, return_distance=False)
dist, ind = kdt.query(query_matrix, k=K)
end = time.time() 
print('kNN time total=%f (sec), per query=%f (sec), per query adjusted for thread number=%f (sec)' % 
      (end-start, float(end-start)/query_qty, num_threads*float(end-start)/query_qty)) 

#reslut[0][0][1]
#print('dist\n', dist)
#print('\n\nind\n', ind)#[0]



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

print('query_qty\n', query_qty)
# Finally computing recall
recall=0.0
for i in range(0, query_qty):
  correct_set = set(gs[1][i])
  #print('gs[1][i]\n', gs[1][i])
  #ret_set = set(nbrs[i][0])
  #print('\nind[i]\n', ind[i])
  ret_set = set(ind[i])
  recall = recall + float(len(correct_set.intersection(ret_set))) / len(correct_set)
recall = recall / query_qty
print('kNN recall %f' % recall)


