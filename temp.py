import numpy as np

order = []
for v in range(1,9):
    p = (v-1)*8+1
    q = 120-(v-1)*8+1
    order.append([p,q,p+1,q+1,p+2,q+2,p+3,q+3,p+4,q+4,p+5,q+5,p+6,q+6,p+7,q+7])
order = np.array(order).flatten()

for i in range(128):
    print(order[i]-1)