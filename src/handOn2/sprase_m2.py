import timeit

from numpy import array
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix, dok_matrix, lil_matrix

#create dense matrix
A = array([[1, 0, 0, 1, 0, 0], [0, 0, 2, 0, 0, 1], [0, 0, 0, 2, 0, 0]])
print(A)

#create sparse matrix
S = coo_matrix(A)
print(S)

print(S.tocsr()[:, 3])

# reconstruct dense matrix
B = S.todense()
print(B)

# timeit
times = 100000
# print properly format with seconds
print(f"dok matrix : {timeit.timeit(lambda: dok_matrix(B), number=times) / times:.5f} seconds")
print(f"lil matrix : {timeit.timeit(lambda: lil_matrix(B), nu้ีmber=times) / times:.5f} seconds")
print(f"csr matrix : {timeit.timeit(lambda: csr_matrix(B), number=times) / times:.5f} seconds")
print(f"csc matrix : {timeit.timeit(lambda: csc_matrix(B), number=times) / times:.5f} seconds")


# print(timeit.timeit(lambda: coo_matrix(B), number=times) / times)
