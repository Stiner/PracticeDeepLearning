import sys, os
sys.path.append(os.pardir)
import numpy as np
import pickle
from dataset.mnist import load_mnist
from funcs import sigmoid
from funcs import softmax
from funcs import img_show

def main():
    x, t = get_data()
    network = init_network()

    ## 이미지 뷰어로 보기
    #idx = 9
    #y = predict(network, x[idx])
    #p = np.argmax(y)
    #print(p)
    #img = x[idx].reshape(28, 28)
    #img = img * 255
    #img_show(img)

    accuracy_cnt = 0
    ## 이미지 1장을 계산
    #for i in range(len(x)):
    #    y = predict(network, x[i])
    #    p = np.argmax(y) ## 확율이 가장 높은 원소의 인덱스를 얻는다.
    #    if p == t[i]:
    #        accuracy_cnt += 1

    ## 이미지 10000장을 한번에 계산
    #y = predict(network, x) ## y.shape = (10000, 10)
    #for i in range(len(y)):
    #    p = np.argmax(y[i]) ## 10개의 확율값 중 가장 높은값의 인덱스(이미지 내용이 인덱스의 값일 확율)
    #    accuracy_cnt += np.sum(p == t[i])

    ## 이미지 100장씩 나누어서 계산
    batch_size = 100
    for i in range(0, len(x), batch_size):
        x_batch = x[i:i+batch_size]
        y_batch = predict(network, x_batch)
        p = np.argmax(y_batch, axis=1)
        accuracy_cnt += np.sum(p == t[i:i+batch_size])

    print("Accuracy:" + str(float(accuracy_cnt) / len(x)))

def get_data(): ## (시험 이미지, 시험 라벨)
    (x_train, t_train), (x_test, t_test) = \
        load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test

def get_train(): ## (훈련 이미지, 훈련 라벨)
    (x_train, t_train), (x_test, t_test) = \
        load_mnist(normalize=False, flatten=True, one_hot_label=False)
    return x_train, t_train

def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)

    return network

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y
