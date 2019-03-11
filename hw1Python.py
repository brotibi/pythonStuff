import numpy as np

A = [[0.1, 0.2, 0.7],
     [0.7, 0.1, 0.2],
     [0.1, 0.3, 0.6]]

B = [[0.2, 0.4, 0.4],
     [0.3, 0.4, 0.3],
     [0.4, 0.3, 0.3]]


c = [1, 0, 0]
d = [0.3, 0.4, 0.3]
e = [1, 1, 1]




def matrixN(matrix, vector, n):
    for i in range(n):
        vector = np.matmul(matrix, vector)
    return vector
        

matrixN(A, c, 100) #[0.1, 0.7, 0.1]
matrixN(A, d, 100) #[0.32, 0.31, 0.33])
matrixN(A, e, 100) #[1., 1., 1.]
matrixN(B, c, 100) #[0.2, 0.3, 0.4]
matrixN(B, d, 100) #[0.34, 0.34, 0.33]
matrixN(B, e, 100) #[1., 1., 1.]

#ii Response no they don't
#yes they are


    

