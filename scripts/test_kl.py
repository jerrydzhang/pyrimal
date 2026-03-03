import numpy as np
import matplotlib.pyplot as plt
data = np.loadtxt("data/zhong/f01/f01.csv", delimiter=",")
print("Min/Max X0:", np.min(data[:,0]), np.max(data[:,0]))
print("Min/Max X1:", np.min(data[:,1]), np.max(data[:,1]))
