import logging
logging.basicConfig(level=logging.NOTSET) #WARNING  NOTSET
#logging.basicConfig(level=logging.ERROR) DEBUG


# Only log WARNING messages and above from nmslib DEBUG logging.NOTSET
#logging.getLogger('nmslib').setLevel(logging.NOTSET)

import nmslib
import numpy

#logger = logging.getLogger(nmslib)

# create a random matrix to index
data = numpy.random.randn(10000, 100).astype(numpy.float32)



# initialize a new index, using a HNSW index on Cosine Similarity
index = nmslib.init(method='hnsw', space='cosinesimil')
index.addDataPointBatch(data)
index.createIndex({'post': 2}, print_progress=True)

# query for the nearest neighbours of the first datapoint
ids, distances = index.knnQuery(data[0], k=10)

# get all nearest neighbours for all the datapoint
# using a pool of 4 threads to compute
neighbours = index.knnQueryBatch(data, k=10, num_threads=4)
