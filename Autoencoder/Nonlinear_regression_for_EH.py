import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
import pickle 
import scipy.io

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

def Initialize_parameters(): # n is the number of channel use
    # Encoder:
    b = 5
    W1 = tf.get_variable("W1", [2*b, 1], initializer = tf.contrib.layers.xavier_initializer())
    b1 = tf.get_variable("b1", [2*b, 1], initializer = tf.zeros_initializer())
    W2 = tf.get_variable("W2", [b, 2*b], initializer = tf.contrib.layers.xavier_initializer())
    b2 = tf.get_variable("b2", [b, 1], initializer = tf.zeros_initializer())
    W3 = tf.get_variable("W3", [b, b], initializer = tf.contrib.layers.xavier_initializer())
    b3 = tf.get_variable("b3", [b, 1], initializer = tf.zeros_initializer())
    W4 = tf.get_variable("W4", [b, b], initializer = tf.contrib.layers.xavier_initializer())
    b4 = tf.get_variable("b4", [b, 1], initializer = tf.zeros_initializer())
    W5 = tf.get_variable("W5", [1, b], initializer = tf.contrib.layers.xavier_initializer())
    b5 = tf.get_variable("b5", [1, 1], initializer = tf.zeros_initializer())

    parameters = {"W1":W1, "b1":b1,
                  "W2":W2, "b2":b2,
                  "W3":W3, "b3":b3,
                  "W4":W4, "b4":b4,
                  "W5":W5, "b5":b5}
    return parameters
"_____________________________________________________________________________"

def forward_propagation(X, parameters, M_mb):
    W1, b1 = parameters['W1'], parameters['b1']
    W2, b2 = parameters['W2'], parameters['b2']
    W3, b3 = parameters['W3'], parameters['b3']
    W4, b4 = parameters['W4'], parameters['b4']
    W5, b5 = parameters['W5'], parameters['b5']

    Z1 = tf.matmul(W1,X)+b1
    A1 = tf.nn.tanh(Z1)
    Z2 = tf.matmul(W2,A1)+b2    
    A2 = tf.nn.tanh(Z2)
    Z3 = tf.matmul(W3,A2)+b3
    A3 = tf.nn.tanh(Z3)
    Z4 = tf.matmul(W4,A3)+b4
    A4 = tf.nn.tanh(Z4)
    Z5 = tf.matmul(W5,A4)+b5
    A5 = tf.nn.tanh(Z5)
    
    return A5
"_____________________________________________________________________________"

Batch_size = 1151
max_itr = 1000
num_epochs = 50000
learning_rate = 0.01
model = scipy.io.loadmat("Model.mat", mdict=None)
Model = model["Model"]
P_in = Model[:,0].reshape(1,Batch_size)
P_out = Model[:,1].reshape(1,Batch_size)
plt.plot(P_in,P_out,'.b')
plt.show()

"Defining the system model"
ops.reset_default_graph()  
X = tf.placeholder(tf.float32, [1,None])
M_mb = tf.placeholder(tf.int64)
parameters = Initialize_parameters()
A5 = forward_propagation(X, parameters, M_mb)
cost = tf.reduce_mean((A5 - P_out)**2)
optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08, name='Adam').minimize(cost)
   
"Optimization"
X_test = np.arange(-40,100,0.1)
X_test = X_test.reshape(1,len(X_test))
with tf.Session() as sess:
    
    Sys_params = []
    cost_min = 1000.
    for i in range(max_itr):
        print(i)
        learning_rate = 0.01
        init = tf.global_variables_initializer()
        costs = []
        cost1=1000.
        sess.run(init)
        for epoch in range(num_epochs):
            _ , minibatch_cost = sess.run([optimizer, cost],  feed_dict={X: P_in, M_mb: Batch_size})
            
            costs.append(minibatch_cost)
            if epoch!= 0 and epoch % 10 == 0:
                cost2 = np.mean(costs[-50::])
                if cost2 < cost1 :
                    cost1 = np.copy(cost2)
                else:
                    learning_rate/=1.01                
                    
                    
            if learning_rate<0.00000000000000000001 or np.isnan(minibatch_cost):
                #print('Optimization break because of small learning rate or cost = nan')
                #print("The learning rate is: ", learning_rate)
                break
            
        if minibatch_cost < cost_min and np.isnan(minibatch_cost)!=1:
            cost_min = minibatch_cost
            Sys_params = sess.run(parameters)
            "Save the file"
            file_name = "Sys_params.pickle"
            with open(file_name, 'wb') as f:
                pickle.dump(Sys_params, f)
            print(minibatch_cost)
            "Obtaining the constellaitons and the cost"
            Y_test = sess.run(A5, feed_dict={X: X_test, M_mb:X_test.shape[1]})
         
    plt.plot(X_test,Y_test,'r.-')
    plt.plot(P_in,P_out,'b.')
    plt.show()
    
    print(cost_min)
    
    plt.plot(X_test,Y_test,'r.-')
    plt.plot(P_in,P_out,'b.')
    plt.axis([-10, 10, 0, 0.09])
    plt.show()
        
sess.close()

    
###############################################################################
###############################################################################
###############################################################################
###############################################################################

"Testimg the error of the solution"
model = scipy.io.loadmat("Model.mat", mdict=None)
Model = model["Model"]
P_in = Model[:,0].reshape(1,1151)
P_out = Model[:,1].reshape(1,1151)

"open the EH_Model"
file_name = 'sys_params.pickle'
with open(file_name, 'rb') as f:
    EH_Model = pickle.load(f)

ops.reset_default_graph()  
X = tf.placeholder(tf.float32, [1,None])
M_mb = tf.placeholder(tf.int64)
A5 = forward_propagation(X, EH_Model, M_mb)
cost = tf.reduce_mean((A5 - P_out)**2)

with tf.Session() as sess:
    print(sess.run(cost, feed_dict={X: P_in, M_mb:1151}))
sess.close()

    
"the mean squarred error is 1.3731867e-10"


"open the EH_Model"
file_name = 'Sys_params.pickle'
with open(file_name, 'rb') as f:
    EH_Model = pickle.load(f)
    
model = scipy.io.loadmat("Model.mat", mdict=None)
Model = model["Model"]
P_in = Model[:,0].reshape(1,1151)
P_out = Model[:,1].reshape(1,1151)
        
W1, b1 = EH_Model['W1'], EH_Model['b1']
W2, b2 = EH_Model['W2'], EH_Model['b2']
W3, b3 = EH_Model['W3'], EH_Model['b3']
W4, b4 = EH_Model['W4'], EH_Model['b4']
W5, b5 = EH_Model['W5'], EH_Model['b5']

X = tf.placeholder(tf.float32, [1,None])
Z1 = tf.matmul(W1,X)+b1
A1 = tf.nn.tanh(Z1)
Z2 = tf.matmul(W2,A1)+b2    
A2 = tf.nn.tanh(Z2)
Z3 = tf.matmul(W3,A2)+b3
A3 = tf.nn.tanh(Z3)
Z4 = tf.matmul(W4,A3)+b4
A4 = tf.nn.tanh(Z4)
Z5 = tf.matmul(W5,A4)+b5
del_power = tf.nn.tanh(Z5)

x_test = np.arange(-40,100,0.1).reshape(1,1400)

with tf.Session() as sess:
    p_del = sess.run(del_power, feed_dict={X: x_test})
sess.close()
    
#plt.plot(np.squeeze(x_test),np.squeeze(10**(x_test/10)),'b')
plt.plot(np.squeeze(x_test),np.squeeze(p_del),'b')
#plt.plot(np.squeeze(P_in),np.squeeze(P_out),'o')
plt.axis([-20, 10, 0, 0.09])
plt.show()





