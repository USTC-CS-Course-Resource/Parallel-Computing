import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.function_base import _quantile_is_valid

# mat vec
x = np.array([1024,2048,4096,8192,16384])
matvec_cpu_speed = 1 / np.array([2239,	10986	,44837	,157256	,557032])
matvec_gpu_speed = 1 / np.array([92318,	95560	,117725	,209020	,520497])
plt.plot(x, 1 / matvec_cpu_speed, c='cyan',label='cpu time cost')
plt.plot(x, 1 / matvec_gpu_speed, c='lawngreen',label='gpu time cost')
plt.scatter(x, 1/matvec_cpu_speed, c='cyan')
plt.scatter(x, 1/matvec_gpu_speed, c='lawngreen')
plt.xlabel('#matrix size')
plt.ylabel(r'time cost')
plt.legend()
plt.show()

# mat mat
x = np.array([128,256,512,1024,2048])
matvec_cpu_speed = 1 / np.array([4323,	48267,	389107,	3599929,	97400035])
matvec_gpu_speed = 1 / np.array([90087,	97075,	129704,	378168,	3635020])

plt.plot(x, 1 / matvec_cpu_speed, c='cyan',label='cpu')
plt.plot(x, 1 / matvec_gpu_speed, c='lawngreen',label='gpu')
plt.scatter(x, 1/matvec_cpu_speed, c='cyan')
plt.scatter(x, 1/matvec_gpu_speed, c='lawngreen')
plt.xlabel('#matrix size')
plt.ylabel(r'time cost')
plt.legend()
plt.show()

# mat mat
x = np.array([128,256,512,1024,2048])
matvec_cpu_speed = 1 / np.array([4323,	48267,	389107,	3599929,	97400035])
matvec_gpu_speed = 1 / np.array([90087,	97075,	129704,	378168,	3635020])
matvec_gpu_shared_speed = 1 / np.array([87976,	92704,	101089,	111836,	1892106])

plt.plot(x, 1 / matvec_cpu_speed, c='cyan',label='cpu')
plt.plot(x, 1 / matvec_gpu_speed, c='lawngreen',label='gpu')
plt.plot(x, 1 / matvec_gpu_shared_speed, c='deeppink',label='gpu shared memory')
plt.scatter(x, 1/matvec_cpu_speed, c='cyan')
plt.scatter(x, 1/matvec_gpu_speed, c='lawngreen')
plt.plot(x, 1 / matvec_gpu_shared_speed, c='deeppink',label='gpu shared memory')
plt.xlabel('#matrix size')
plt.ylabel(r'time cost')
plt.legend()
plt.show()