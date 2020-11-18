import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import tensorflow as tf
import numpy as np
import math 
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
#import scipy.special as sps
import pickle 

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

def Create_placeholders(nx, ny1, ny2):
    X = tf.placeholder(tf.float32, [nx,None])
    Y1 = tf.placeholder(tf.float32, [ny1,None])
    Y2 = tf.placeholder(tf.float32, [ny2,None])
    return X, Y1, Y2
"_____________________________________________________________________________"
def Initialize_parameters(nx, n, ny1, ny2): # n is the number of channel use
    # Encoder:
    n_h1_en = np.floor(0.5*(nx + n))
    n_h2_en = np.floor(0.5*(n_h1_en + n))
    W1_1 = tf.get_variable("W1_1", [n_h1_en, nx], initializer = tf.contrib.layers.xavier_initializer())
    b1_1 = tf.get_variable("b1_1", [n_h1_en, 1], initializer = tf.zeros_initializer())
    W1_2 = tf.get_variable("W1_2", [n_h1_en, nx], initializer = tf.contrib.layers.xavier_initializer())
    b1_2 = tf.get_variable("b1_2", [n_h1_en, 1], initializer = tf.zeros_initializer())
    W2_1 = tf.get_variable("W2_1", [n_h2_en, 2*n_h1_en], initializer = tf.contrib.layers.xavier_initializer())
    b2_1 = tf.get_variable("b2_1", [n_h2_en, 1], initializer = tf.zeros_initializer())
    W2_2 = tf.get_variable("W2_2", [n_h2_en, 2*n_h1_en], initializer = tf.contrib.layers.xavier_initializer())
    b2_2 = tf.get_variable("b2_2", [n_h2_en, 1], initializer = tf.zeros_initializer())
    W3 = tf.get_variable("W3", [n, 2*n_h2_en], initializer = tf.contrib.layers.xavier_initializer())
    b3 = tf.get_variable("b3", [n, 1], initializer = tf.zeros_initializer())
    
    # Decoder1:
    n_h1_de1 = np.floor(0.5*(n + ny1))
    n_h2_de1 = np.floor(0.5*(n_h1_de1 + ny1))
    W41_1 = tf.get_variable("W41_1", [n_h1_de1, n], initializer = tf.contrib.layers.xavier_initializer())
    b41_1 = tf.get_variable("b41_1", [n_h1_de1, 1], initializer = tf.zeros_initializer())
    W41_2 = tf.get_variable("W41_2", [n_h1_de1, n], initializer = tf.contrib.layers.xavier_initializer())
    b41_2 = tf.get_variable("b41_2", [n_h1_de1, 1], initializer = tf.zeros_initializer())
    W51_1 = tf.get_variable("W51_1", [n_h2_de1, 2*n_h1_de1], initializer = tf.contrib.layers.xavier_initializer())
    b51_1 = tf.get_variable("b51_1", [n_h2_de1, 1], initializer = tf.zeros_initializer())
    W51_2 = tf.get_variable("W51_2", [n_h2_de1, 2*n_h1_de1], initializer = tf.contrib.layers.xavier_initializer())
    b51_2 = tf.get_variable("b51_2", [n_h2_de1, 1], initializer = tf.zeros_initializer())
    W61 = tf.get_variable("W61", [ny1, 2*n_h2_de1], initializer = tf.contrib.layers.xavier_initializer())
    b61 = tf.get_variable("b61", [ny1, 1], initializer = tf.zeros_initializer())
    
    # Decoder2:
    n_h1_de2 = np.floor(0.5*(n + ny2))
    n_h2_de2 = np.floor(0.5*(n_h1_de2 + ny2))
    W42_1 = tf.get_variable("W42_1", [n_h1_de2, n], initializer = tf.contrib.layers.xavier_initializer())
    b42_1 = tf.get_variable("b42_1", [n_h1_de2, 1], initializer = tf.zeros_initializer())
    W42_2 = tf.get_variable("W42_2", [n_h1_de2, n], initializer = tf.contrib.layers.xavier_initializer())
    b42_2 = tf.get_variable("b42_2", [n_h1_de2, 1], initializer = tf.zeros_initializer())
    W52_1 = tf.get_variable("W52_1", [n_h2_de2, 2*n_h1_de2], initializer = tf.contrib.layers.xavier_initializer())
    b52_1 = tf.get_variable("b52_1", [n_h2_de2, 1], initializer = tf.zeros_initializer())
    W52_2 = tf.get_variable("W52_2", [n_h2_de2, 2*n_h1_de2], initializer = tf.contrib.layers.xavier_initializer())
    b52_2 = tf.get_variable("b52_2", [n_h2_de2, 1], initializer = tf.zeros_initializer())
    W62 = tf.get_variable("W62", [ny2, 2*n_h2_de2], initializer = tf.contrib.layers.xavier_initializer())
    b62 = tf.get_variable("b62", [ny2, 1], initializer = tf.zeros_initializer())
    
    parameters = {"W1_1":W1_1, "b1_1":b1_1, "W1_2":W1_2, "b1_2":b1_2,
                  "W2_1":W2_1, "b2_1":b2_1, "W2_2":W2_2, "b2_2":b2_2,
                  "W3":W3, "b3":b3, 
                  "W41_1":W41_1, "b41_1":b41_1, "W41_2":W41_2, "b41_2":b41_2,
                  "W51_1":W51_1, "b51_1":b51_1, "W51_2":W51_2, "b51_2":b51_2,
                  "W61":W61, "b61":b61, 
                  "W42_1":W42_1, "b42_1":b42_1, "W42_2":W42_2, "b42_2":b42_2,
                  "W52_1":W52_1, "b52_1":b52_1, "W52_2":W52_2, "b52_2":b52_2,
                  "W62":W62, "b62":b62}
    return parameters
"_____________________________________________________________________________"
def forward_propagation(X, parameters, avrg_pwr, noise_std1, noise_std2, M_mb):
    #TX
    W1_1, b1_1 = parameters['W1_1'], parameters['b1_1']
    W1_2, b1_2 = parameters['W1_2'], parameters['b1_2']
    W2_1, b2_1 = parameters['W2_1'], parameters['b2_1']
    W2_2, b2_2 = parameters['W2_2'], parameters['b2_2']
    W3, b3 = parameters['W3'], parameters['b3']
    #RX1
    W41_1, b41_1 = parameters['W41_1'], parameters['b41_1'] 
    W41_2, b41_2 = parameters['W41_2'], parameters['b41_2']
    W51_1, b51_1 = parameters['W51_1'], parameters['b51_1']
    W51_2, b51_2 = parameters['W51_2'], parameters['b51_2']
    W61, b61 = parameters['W61'], parameters['b61']
    #RX2
    W42_1, b42_1 = parameters['W42_1'], parameters['b42_1'] 
    W42_2, b42_2 = parameters['W42_2'], parameters['b42_2']
    W52_1, b52_1 = parameters['W52_1'], parameters['b52_1']
    W52_2, b52_2 = parameters['W52_2'], parameters['b52_2']
    W62, b62 = parameters['W62'], parameters['b62']
    
    Z1_1 = tf.matmul(W1_1,X) + b1_1
    Z1_2 = tf.matmul(W1_2,X) + b1_2
    A1 = tf.concat([tf.nn.relu(Z1_1),tf.nn.tanh(Z1_2)],0)
    Z2_1 = tf.matmul(W2_1,A1) + b2_1
    Z2_2 = tf.matmul(W2_2,A1) + b2_2    
    A2 = tf.concat([tf.nn.relu(Z2_1),tf.nn.tanh(Z2_2)],0)  
    Z3 = tf.matmul(W3,A2)+b3
    A3 = tf.nn.tanh(Z3) 
    # Normalization of the channel input
    A3 = tf.div(A3, tf.sqrt( tf.reduce_mean( tf.reduce_sum(tf.multiply(A3,A3),axis=0) ))) * np.sqrt(avrg_pwr)
    
    # Received signals
    Yr1 = A3 + tf.random_normal([A3.shape[0],M_mb],mean=0.0,stddev=noise_std1)
    Yr2 = A3 + tf.random_normal([A3.shape[0],M_mb],mean=0.0,stddev=noise_std2)
    
    # NN at receiver 1
    Z41_1 = tf.matmul(W41_1,Yr1)+b41_1
    Z41_2 = tf.matmul(W41_2,Yr1)+b41_2
    A41 = tf.concat([tf.nn.relu(Z41_1),tf.nn.tanh(Z41_2)],0)
    Z51_1 = tf.matmul(W51_1,A41)+b51_1
    Z51_2 = tf.matmul(W51_2,A41)+b51_2
    A51 = tf.concat([tf.nn.relu(Z51_1),tf.nn.tanh(Z51_2)],0)
    Z61 = tf.matmul(W61,A51)+b61
    
    # NN at receiver 2
    Z42_1 = tf.matmul(W42_1,Yr2)+b42_1
    Z42_2 = tf.matmul(W42_2,Yr2)+b42_2
    A42 = tf.concat([tf.nn.relu(Z42_1),tf.nn.tanh(Z42_2)],0)
    Z52_1 = tf.matmul(W52_1,A42)+b52_1
    Z52_2 = tf.matmul(W52_2,A42)+b52_2
    A52 = tf.concat([tf.nn.relu(Z52_1),tf.nn.tanh(Z52_2)],0)
    Z62 = tf.matmul(W62,A52)+b62
    
    return A3, Yr1, Yr2, Z61, Z62
"_____________________________________________________________________________"
def compute_delivery_power(Rx1, Rx2, EH_Model):
    W1, b1 = EH_Model['W1'], EH_Model['b1']
    W2, b2 = EH_Model['W2'], EH_Model['b2']
    W3, b3 = EH_Model['W3'], EH_Model['b3']
    W4, b4 = EH_Model['W4'], EH_Model['b4']
    W5, b5 = EH_Model['W5'], EH_Model['b5']
    
    P_in1 = 4.342944819032518*tf.log(tf.reduce_sum(Rx1**2,axis=0, keepdims=True))
    P_in2 = 4.342944819032518*tf.log(tf.reduce_sum(Rx2**2,axis=0, keepdims=True))
    
    Z11 = tf.matmul(W1,P_in1)+b1
    A11 = tf.nn.tanh(Z11)
    Z21 = tf.matmul(W2,A11)+b2    
    A21 = tf.nn.tanh(Z21)
    Z31 = tf.matmul(W3,A21)+b3
    A31 = tf.nn.tanh(Z31)
    Z41 = tf.matmul(W4,A31)+b4
    A41 = tf.nn.tanh(Z41)
    Z51 = tf.matmul(W5,A41)+b5
    del_power1 = tf.reduce_mean(tf.nn.tanh(Z51))
    
    Z12 = tf.matmul(W1,P_in2)+b1
    A12 = tf.nn.tanh(Z12)
    Z22 = tf.matmul(W2,A12)+b2    
    A22 = tf.nn.tanh(Z22)
    Z32 = tf.matmul(W3,A22)+b3
    A32 = tf.nn.tanh(Z32)
    Z42 = tf.matmul(W4,A32)+b4
    A42 = tf.nn.tanh(Z42)
    Z52 = tf.matmul(W5,A42)+b5
    del_power2 = tf.reduce_mean(tf.nn.tanh(Z52))
    
    return del_power1+del_power2
"_____________________________________________________________________________"
def compute_cost(Z1, Z2, Y1, Y2, Rx1, Rx2, lmbd, EH_Model):
    W1, b1 = EH_Model['W1'], EH_Model['b1']
    W2, b2 = EH_Model['W2'], EH_Model['b2']
    W3, b3 = EH_Model['W3'], EH_Model['b3']
    W4, b4 = EH_Model['W4'], EH_Model['b4']
    W5, b5 = EH_Model['W5'], EH_Model['b5']
    
    P_in1 = 4.342944819032518*tf.log(tf.reduce_sum(Rx1**2,axis=0, keepdims=True))
    P_in2 = 4.342944819032518*tf.log(tf.reduce_sum(Rx2**2,axis=0, keepdims=True))
    
    Z11 = tf.matmul(W1,P_in1)+b1
    A11 = tf.nn.tanh(Z11)
    Z21 = tf.matmul(W2,A11)+b2    
    A21 = tf.nn.tanh(Z21)
    Z31 = tf.matmul(W3,A21)+b3
    A31 = tf.nn.tanh(Z31)
    Z41 = tf.matmul(W4,A31)+b4
    A41 = tf.nn.tanh(Z41)
    Z51 = tf.matmul(W5,A41)+b5
    del_power1 = tf.reduce_mean(tf.nn.tanh(Z51))
    
    Z12 = tf.matmul(W1,P_in2)+b1
    A12 = tf.nn.tanh(Z12)
    Z22 = tf.matmul(W2,A12)+b2    
    A22 = tf.nn.tanh(Z22)
    Z32 = tf.matmul(W3,A22)+b3
    A32 = tf.nn.tanh(Z32)
    Z42 = tf.matmul(W4,A32)+b4
    A42 = tf.nn.tanh(Z42)
    Z52 = tf.matmul(W5,A42)+b5
    del_power2 = tf.reduce_mean(tf.nn.tanh(Z52))
    
    lgts1 = tf.transpose(Z1)
    lbls1 = tf.transpose(Y1)
    lgts2 = tf.transpose(Z2)
    lbls2 = tf.transpose(Y2)
    CE1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = lgts1, labels = lbls1))
    CE2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = lgts2, labels = lbls2))
#    cost = CE1 + CE2 + lmbd*(1/del_power1 + 1/del_power2)
    cost = (CE1 + CE2) + lmbd/(del_power1 + del_power2)
    return cost
"_____________________________________________________________________________"
def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    m = X.shape[1]                  
    mini_batches = []
    np.random.seed(seed)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation]
    num_complete_minibatches = math.ceil(m/mini_batch_size) 
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size : np.minimum((k+1) * mini_batch_size, m)]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size : np.minimum((k+1) * mini_batch_size, m)]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    return mini_batches
"_____________________________________________________________________________"
def compute_SER(Y, Y_est, nx1, nx2):
    SER = np.sum(np.maximum((Y_est[0:nx1,:].argmax(axis=0)-Y[0:nx1,:].argmax(axis=0))!=0,
                 (Y_est[nx1::,:].argmax(axis=0)-Y[nx1::,:].argmax(axis=0))!=0 ))/(Y.shape[1])
    return SER  
"_____________________________________________________________________________"
def main():
    "Open the EH model"
    file_name = 'Sys_params.pickle'
    with open(file_name, 'rb') as f:
        EH_Model = pickle.load(f)
        
    delta_lmbd = 0.001#0.000005
    lmbd_max = 30
    file_name = 'BC_nx1_8_nx2_4_0p1.pickle'
    max_itr= 61
    Simulation = "Middle"
    Number_of_bits1 = 3
    Number_of_bits2 = 2
    nx1 = np.power(2,Number_of_bits1)
    nx2 = np.power(2,Number_of_bits2)
    SER = 0.95
    SER_com = 0.
    av_pwr = .1 # change the noise as well
    noise_std1 = 1/(np.sqrt(600/av_pwr))
    noise_std2 = 1/(np.sqrt(300/av_pwr))
    n = 2
    m_tr = 10000
    m_ts = (nx1+nx2) * 50000
    mb_size = 10000
    seed = 1                 
    num_epochs = 5000
    learning_rate = 0.01
    
    "For the start from the middle of the simulaiton"
    if Simulation == "Middle":
        lmbd = 0.12
    elif Simulation == "Start":
        lmbd = 0.
        
    "Defining the system model"
    ops.reset_default_graph()                         
    X, Y1, Y2 = Create_placeholders(nx1+nx2, nx1, nx2)
    M_mb = tf.placeholder(tf.int64)
    lmbd_ph = tf.placeholder(tf.float32)
    parameters = Initialize_parameters(nx1+nx2, n, nx1, nx2)
    A3, Yr1, Yr2, Z61, Z62 = forward_propagation(X, parameters, av_pwr, noise_std1, noise_std2, M_mb)
    cost = compute_cost(Z61, Z62, Y1, Y2, Yr1, Yr2, lmbd_ph, EH_Model)
    P_del_tf = compute_delivery_power(Yr1, Yr2, EH_Model)
    optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08, name='Adam').minimize(cost)
    
    "Producing the training and test datas"
    X_ts1 = tf.one_hot(tf.squeeze(tf.random.uniform((1,m_ts), 0, nx1, dtype = tf.int32)),depth = nx1 ,axis=0)
    X_ts2 = tf.one_hot(tf.squeeze(tf.random.uniform((1,m_ts), 0, nx2, dtype = tf.int32)),depth = nx2 ,axis=0)
    X_tr1 = tf.one_hot(tf.squeeze(tf.random.uniform((1,m_tr), 0, nx1, dtype = tf.int32)),depth = nx1 ,axis=0)
    X_tr2 = tf.one_hot(tf.squeeze(tf.random.uniform((1,m_tr), 0, nx2, dtype = tf.int32)),depth = nx2 ,axis=0)
    X_ts  = tf.concat([X_ts1, X_ts2], 0)
    X_tr  = tf.concat([X_tr1, X_tr2], 0)
    
    "Optimization"
    with tf.Session() as sess:
        X_train = sess.run(X_tr)
        X_test = sess.run(X_ts)
        while (SER_com <= SER) and lmbd < lmbd_max: 
            Constel, Rx1, Rx2, Y_est1, Y_est2, cst, Sys_params = [], [], [], [], [], [], []
            cost_min = 1000000000.
            for i in range(max_itr):
                learning_rate = 0.01
                init = tf.global_variables_initializer()
                if i%10 == 0:
                    print("i = " + str(i) + " for nx1: "+ str(nx1) + " for nx2: "+ str(nx2) + " lambda: "+ str(lmbd))
                costs = []
                cost1=1000000000.
                sess.run(init)
                for epoch in range(num_epochs):
                    epoch_cost = 0.                       # Defines a cost related to an epoch
                    num_minibatches = math.ceil(m_tr / mb_size) # number of minibatches of size mb_size in the train set
                    seed += 1
                    minibatches = random_mini_batches(X_train, X_train, mb_size, seed)
                    for minibatch in minibatches:
                        (minibatch_X, minibatch_Y) = minibatch
                        _ , minibatch_cost = sess.run([optimizer, cost],  feed_dict={X: minibatch_X, Y1: minibatch_Y[0:nx1,:], Y2: minibatch_Y[nx1::,:],
                                                                                     M_mb: minibatch_X.shape[1], lmbd_ph: lmbd})
                        epoch_cost += minibatch_cost / num_minibatches
#        
                    costs.append(epoch_cost)
                    if epoch!= 0 and epoch % 60 == 0:
                        cost2 = np.mean(costs[-50::])
                        if cost2 < cost1 :
                            cost1 = np.copy(cost2)
                        else:
                            learning_rate/=1.01                
        
                    if learning_rate<0.00001 or np.isnan(epoch_cost):
                        print('Optimization break because of small learning rate or cost = nan')
                        break
#                
                if epoch_cost < cost_min:
                    cost_min = epoch_cost
                    Sys_params = sess.run(parameters)
                    "Obtaining the constellaitons and the cost"
                    Constel, Rx1, Rx2, Y_est1, Y_est2, cst = sess.run([A3, Yr1, Yr2, Z61, Z62, cost],
                                                                      feed_dict={X: X_test,
                                                                                 Y1: X_test[0:nx1,:], Y2: X_test[nx1::,:],
                                                                                 M_mb: X_test.shape[1], lmbd_ph: lmbd})                    
            Y_est = np.concatenate((Y_est1, Y_est2), axis=0)
            SER_com = compute_SER(X_test, Y_est, nx1, nx2)
                    
            P_del = sess.run(P_del_tf, feed_dict={Yr1: Rx1, Yr2: Rx2})
            print(lmbd, SER_com, P_del, cst)
            
            plt.plot(Constel[0,0:1000],Constel[1,0:1000],'r.')
            plt.axis("equal")
            plt.show()
            
            if Simulation == "Middle":
                "Open the file"
                with open(file_name, 'rb') as f:
                    Optimized_params = pickle.load(f)
            elif Simulation == "Start":
                Optimized_params = []
                Simulation = "Middle"
                
            Optimized_params += [[nx1, nx2, lmbd, av_pwr, noise_std1, noise_std2, SER_com, P_del, cst, Sys_params, Constel[:,0:1000]]]
            lmbd += delta_lmbd+lmbd*0.04
            print(lmbd)
            "SAVE (http://www.jessicayung.com/how-to-use-pickle-to-save-and-load-variables-in-python/)"
            with open(file_name, 'wb') as f:
                pickle.dump(Optimized_params, f)
            
    sess.close()
    
main()     


#i=3
#plt.plot(Optimized_params[i][10][0,:],Optimized_params[i][10][1,:],'r.')
#plt.axis("equal")
#plt.show()

