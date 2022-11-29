import numpy as np

info_txt = open('F:\datasets\Diverse_Masked_Faces_v2_m/new_cat_info.txt', 'r')


lines = info_txt.readlines()
a = []
for l in lines:
    b = l[11:-3].replace(' ', '').split('.')
    # c = [int(x) for x in b]
    c = list(map(int, b))
    # if np.array(c)[:2]==0:
    a.append(c)
d = np.array(a)
print(b)