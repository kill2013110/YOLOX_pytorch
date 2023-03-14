import numpy as np
from matplotlib import pyplot as plt

c
x1 = np.random.normal(1.75, 1, 100000000)
# 画图看分布状况
# 1）创建画布
plt.figure(figsize=(20, 10), dpi=100)
# 2）绘制直方图
plt.hist(x1, 1000)
# 3）显示图像
plt.show()
