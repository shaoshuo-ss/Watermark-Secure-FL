# -*- coding: UTF-8 -*-
import os

for i in range(500 // 50):
    os.system("python pmain.py >> cifar10-iid-VGG-16-baseline.txt")
    print("Number:%d finished" % i)
