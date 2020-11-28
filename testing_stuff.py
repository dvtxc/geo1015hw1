import numpy as np

foo = np.array([[3 ,5 ,6],[2, 4,762],[65,543,2],[34,4,0]])
print(foo)
print(np.where(foo > 10))
print(foo[np.where(foo > 10)])

a = np.array([[3, 4, 0],[2, 6, 2], [ -1, 6, 10], [-7,-7,100]])
b = np.array([[0, 0], [0, 5], [5, 0], [5, 5], [10, 0], [10, 5]])

print(a)
print(b)

dx = np.subtract.outer(b[:,0], a[:,0])
dy = np.subtract.outer(b[:,1], a[:,1])

print(dx)
print('\n')
print(dy)
print('\n')

arr_dist = np.hypot(dx, dy)

print(arr_dist)

print(np.where(arr_dist < 5))
print(arr_dist[ np.where(arr_dist < 5) ])

num_rp, num_sp = arr_dist.shape

out = np.empty(num_rp)
print(out.shape)

# only calculate for small distances
for i in range(num_rp):
    distances = arr_dist[i, :]
    sp_in_circle = np.where(distances < 5)[0]
    
    #if sp_in_circle.size != 0:
    distances = distances[sp_in_circle]
    weights = distances**-2
    values = a[sp_in_circle, 2]

    print(arr_dist[i, :])
    print(distances)
    print(values)
    print('\n')