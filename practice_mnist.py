import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from funcs import img_show

def main():
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

    #print(x_train.shape)
    #print(t_train.shape)
    #print(x_test.shape)
    #print(t_test.shape)

    img = x_train[0]
    label = t_train[0]
    print(label) # 5

    print(img.shape)            # (784, )
    img = img.reshape(28, 28)   # 원래 이미지의 모양으로 변형
    print(img.shape)            # (28, 28)

    img_show(img)               # 이미지 뷰어 실행