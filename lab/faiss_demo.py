import numpy as np
import faiss

d = 64                           # dimension
nb = 100000                      # database size
nq = 10000                       # nb of queries
np.random.seed(1234)             # make reproducible
xb = np.random.random((nb, d)).astype('float32')
xb[:, 0] += np.arange(nb) / 1000.
xq = np.random.random((nq, d)).astype('float32')
xq[:, 0] += np.arange(nq) / 1000.

index = faiss.IndexFlatL2(d)   # build the index
print(index.is_trained)
index.add(xb)                  # add vectors to the index
print(index.ntotal)


# Create an IVF index
nlist = 100  # Number of clusters
quantizer = faiss.IndexFlatL2(64)  # This is the coarse quantizer
index_ivf = faiss.IndexIVFFlat(quantizer, 64, nlist)

# Train the index with your database vectors
index_ivf.train(xb)  # `xb` is your database of vectors
index_ivf.add(xb)  # Add vectors to the index

# Perform range search (returns all vectors within a specific distance)
lim = 0.5  # Distance threshold
query_vector = np.random.random(64).astype('float32')
distances, indices = faiss.RangeSearchResult(1000)
index_ivf.range_search(query_vector.reshape(1, -1), lim, distances, indices)