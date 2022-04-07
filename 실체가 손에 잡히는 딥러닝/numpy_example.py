# numpy import
import numpy_example as np

# numpy array
""""
_1D = np.array([0,1,2,3,4,5])
print(_1D)

_2D = np.array([[0,1,2],[3,4,5]])
print(_2D)

_3D = np.array([[[0,1,2],[3,4,5]],[[5,4,3],[2,1,0]]])
print(_3D)

print('np.shape()', np.shape(_3D))
print('np.size()', np.size(_3D))

arr = [[1,2],[3,4],[5,6]]
print(len(arr))
print(len(np.array(arr)))
"""

# numpy array creating function
"""
print(np.zeros(10))
print(np.ones(10))
print(np.random.rand(10))

print(np.zeros((2,3)))
print(np.ones((3,4,5)))

print(np.arange(0,1,0.1))
print(np.arange(10))

print(np.linspace(0,1,11))
print(np.linspace(1,50))
"""

# reshape
"""
a = np.array([1,2,3,4,5,6,7,8])
print(a.reshape((2,4)))

b = np.reshape(a,(2,4))
print(b)

c = b.reshape(2,2,2)
print(c)

d = c.reshape(4,2)
print(d)

e = d.reshape(-1)
print(e)

f = e.reshape(2,-1)
print(f)
"""

# array operation
"""
a = np.array([0,1,2,3,4,5]).reshape(2,3)
print('a\n',a)
print('a+3\n',a+3)
print('a*3\n',a*3)

b = np.array([5,4,3,2,1,0]).reshape(2,3)
print('b\n',a)
print('a+b\n',a+b)
print('a*b\n',a*b)
"""

# Broadcasting
"""
a = np.array([[1,2],[1,2]])
b = np.array([1,2])
c = np.array([[1],[1]])
print(a+b)
print(a+c)
"""

# index
"""
d = np.array([0,1,2,3,4,5,6,7,8,9])
print(d[d % 2 == 0])

e = np.zeros((3,3))
f = np.array([8,9])
e[np.array([0,2]), np.array([0,1])] = f
print(e)
"""

# slicing
"""
a = np.array([0,1,2,3,4,5,6,7,8,9])
print(a[2:7])
print(a[:])

b = np.array([[0,1,2],
              [3,4,5],
              [6,7,8]])
print(b[0:2, 0:2])

c = np.zeros(18).reshape(2,3,3)
print(c)
c[0, 0:2, 0:2] = np.ones(4).reshape(2,2)
print(c)
"""

# axis and transpose
"""
a = np.array([[0,1,2],
             [3,4,5]])
print(a)
b = a.transpose(1,0)
print(b)
c = b.transpose(1,0)
print(c)

print(a.T)

a = np.arange(12).reshape(2,2,3)
print(a)
b = a.transpose(1,2,0)
print(b)
"""

# numpy function
"""
a = np.array([[0,1],[2,3]])
print(np.sum(a))
print(np.sum(a, axis=0))
print(np.sum(a, axis=1))
print(np.sum(a, axis=0, keepdims=True))
print(np.sum(a, axis=1, keepdims=True))

print(np.max(a))

print(np.argmax(a))
print(np.argmax(a, axis=1))

print(np.where(a<2, 9, a))
"""
