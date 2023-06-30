import logging
logging.basicConfig(level=logging.NOTSET) #WARNING  NOTSET
#logging.basicConfig(level=logging.ERROR) DEBUG


# Only log WARNING messages and above from nmslib DEBUG logging.NOTSET
#logging.getLogger('nmslib').setLevel(logging.NOTSET)


import nmslib
import numpy 

#logger = logging.getLogger(nmslib)

# create a random matrix to index
#data = numpy.random.randn(5, 3).astype(numpy.float32)

#print(data[0])

with open(r"/home/ubuntu/lulingling/testnmslib/final128_10K.txt","r") as f:
   all_data = f.readlines()
   #print( len(numpy.loadtxt(all_data))  )
   data = numpy.loadtxt(all_data)



# initialize a new index, using a HNSW index on Cosine Similarity
index = nmslib.init(method='hnsw', space='cosinesimil')
index.addDataPointBatch(data)
#index.createIndex({'post': 2}, print_progress=True,'M': 32, 'indexThreadQty': 1, 'efConstruction': 800)
index.createIndex({'post': 2, 'M': 32, 'indexThreadQty': 1, 'efConstruction': 800}, print_progress=True)

# query for the nearest neighbours of the first datapoint
#ids, distances = index.knnQuery(data[0], k=10)

# # get all nearest neighbours for all the datapoint
# # using a pool of 4 threads to compute
neighbours = index.knnQueryBatch(data[0:20], k=10, num_threads=1)
