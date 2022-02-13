import numpy as np

x0 = np.matrix([1/7] * 7)
P = np.matrix([
                [1/7, 1/7, 1/7, 1/7, 1/7, 1/7, 1/7],
                [25/56, 3/140, 25/56, 3/140, 3/140, 3/140, 3/140],
                [3/140, 3/140, 3/140, 3/140, 61/70, 3/140, 3/140],
                [3/140, 3/140, 25/56, 3/140, 3/140, 3/140, 25/56],
                [25/56, 3/140, 3/140, 3/140, 3/140, 25/56, 3/140],
                [3/140, 3/140, 61/70, 3/140, 3/140, 3/140, 3/140],
                [3/140, 3/140, 25/56, 3/140, 3/140, 25/56, 3/140],
            ])
print(x0*P)
print(x0 * P * P)
print(x0 * P * P * P)
print(' ')
prev_Px = x0
Px = x0*P
i=0
while(any(abs(np.asarray(prev_Px).flatten()-np.asarray(Px).flatten()) > 1e-8)):
    i+=1
    prev_Px = Px
    Px = Px * P
print('Converged in {0} iterations: {1}'.format(i, np.asarray(Px).flatten()))
