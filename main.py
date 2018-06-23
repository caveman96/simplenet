import numpy as np
import random
import pickle
import gzip
from scipy import misc
from numpy import array


sizes = [784,40,10]
num_layers = len(sizes)
biases = [np.random.randn(y, 1) for y in sizes[1:]]
weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]


def predict(inp):
    for b, w in zip(biases,weights):
        inp = sigmoid(np.dot(w, inp) + b)
    return inp


def g_descent(training_data, epochs, mini_batch_size, learning_rate,test_data):
    n_test = len(test_data)
    n = len(training_data)
    for j in xrange(epochs):
        random.shuffle(training_data)
        mini_batches = [training_data[k:k + mini_batch_size] for k in xrange(0, n, mini_batch_size)]
        for mini_batch in mini_batches:
            update_mini_batch(mini_batch, learning_rate)
        if test_data:
            print ("Epoch {0}: {1} / {2}".format(j+1, evaluate(test_data),n_test))


def update_mini_batch(mini_batch, learning_rate):
    global biases
    global weights
    del_b = [np.zeros(b.shape) for b in biases]
    del_w = [np.zeros(w.shape) for w in weights]
    for x, y in mini_batch:
        delta_b, delta_w = backprop(x, y)
        del_b = [nb + dnb for nb, dnb in zip(del_b, delta_b)]
        del_w = [nw + dnw for nw, dnw in zip(del_w, delta_w)]
    weights = [w - (learning_rate / len(mini_batch)) * nw for w, nw in zip(weights, del_w)]
    biases = [b - (learning_rate / len(mini_batch)) * nb for b, nb in zip(biases, del_b)]


def backprop(x,y):
    del_b=[np.zeros(b.shape) for b in biases]
    del_w=[np.zeros(w.shape) for w in weights]
    activation=x
    activations=[x]
    zs=[]
    for b,w in zip(biases,weights):
        z=np.dot(w,activation)+b
        zs.append(z)
        activation=sigmoid(z)
        activations.append(activation)
    delta=cost_derivative(activations[-1],y)*sigmoid_derivative(zs[-1])
    del_b[-1] = delta
    del_w[-1] = np.dot(delta, activations[-2].transpose())
    for i in xrange(2, num_layers):
        z = zs[-i]
        sp = sigmoid_derivative(z)
        delta = np.dot(weights[-i + 1].transpose(), delta) * sp
        del_b[-i] = delta
        del_w[-i] = np.dot(delta, activations[-i - 1].transpose())
    return del_b, del_w


def evaluate(test_data):
    test_results=[(np.argmax(predict(x)),y) for (x,y) in test_data]
    return sum(int(x==y) for (x,y) in test_results)


def cost_derivative(output_activations,y):
    return output_activations-y


def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))


def sigmoid_derivative(z):
    return sigmoid(z)*(1-sigmoid(z))


def one_hot(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

def conv(z):
    a=[]
    for i in range(10):
        if i==z:
            a.append([1.])
        else:
            a.append([0.])
    return a


if __name__=='__main__':
    f=gzip.open('mnist.pkl.gz', 'rb')
    tr_d,va_d,te_d=pickle.load(f)
    f.close()
    training_inputs=[np.reshape(x,(784,1)) for x in tr_d[0]]
    training_results=[one_hot(y) for y in tr_d[1]]
    train_data=zip(training_inputs,training_results)
    validation_inputs=[np.reshape(x,(784,1)) for x in va_d[0]]
    validation_data=zip(validation_inputs,va_d[1])
    test_inputs=[np.reshape(x,(784,1)) for x in te_d[0]]
    test_data=zip(test_inputs,te_d[1])

    g_descent(train_data,10,10,2.8,test_data)

    with open("biases","wb") as b:
        np.save(b, biases)
        b.close()
    with open("weights","wb") as b:
        np.save(b, weights)
        b.close()

    inp=0
    for i in range(9):
        a1="./test images/"+str(inp)+".png"
        a=misc.imread(a1,flatten=True)
        a=array(a,dtype=float)
        a=a/a.max()
        """
        t=[]
        for i in a:
            c=[]
            for j in i:
                c.append(1-j)
            t.append(c)
        a=t
        a=array(a)
        """
        a.shape=(784,1)
        inp=inp+1
        #print("{0}--{1}".format(np.argmax(predict(test_inputs[0])),te_d[1][0]))
        print(np.argmax(predict(a)))
