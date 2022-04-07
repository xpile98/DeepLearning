import numpy as np

# import module
import matplotlib.pyplot as plt

# graph
"""
x = np.linspace(0,2*np.pi)
y = np.sin(x)
plt.plot(x,y)
plt.show()
"""

# graph design
"""
x = np.linspace(0,2*np.pi)
y_sin = np.sin(x)
y_cos = np.cos(x)
plt.xlabel('x val')
plt.ylabel('y val')
plt.title('sin / cos')
plt.plot(x,y_sin, label="sin")
plt.plot(x,y_cos, label="cos", linestyle="dashed")
plt.legend()
plt.show()
"""

# scatter
"""    
x_1 = np.random.rand(100) - 1
y_1 = np.random.rand(100)
x_2 = np.random.rand(100)
y_2 = np.random.rand(100)
plt.scatter(x_1,y_1, marker='+')
plt.scatter(x_2,y_2, marker='x')
plt.show()
"""

# image show

img = np.array([[0,1,2,3],
                [4,5,6,7],
                [8,9,10,11],
                [12,13,14,15]])
plt.imshow(img,"gray")
plt.colorbar()
plt.show()

img = plt.imread("Symbol.png")
plt.imshow(img)
plt.show()
