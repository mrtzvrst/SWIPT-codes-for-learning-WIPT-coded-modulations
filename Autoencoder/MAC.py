import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import tensorflow as tf
import numpy as np
import math 
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
#import scipy as sp
#import scipy.special as sps
import pickle 

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

def Create_placeholders(nx1, nx2, ny):
    X1 = tf.placeholder(tf.float32, [nx1,None])
    X2 = tf.placeholder(tf.float32, [nx2,None])
    Y = tf.placeholder(tf.float32, [ny,None])
    return X1, X2, Y
"_____________________________________________________________________________"
def Initialize_parameters(nx1, nx2, n, ny): # n is the number of channel use
# Encoder1:
    n_h1_en1 = np.floor(0.5*(nx1 + n))
    n_h2_en1 = np.floor(0.5*(n_h1_en1 + n))
    W11_1 = tf.get_variable("W11_1", [n_h1_en1, nx1], initializer = tf.contrib.layers.xavier_initializer())
    b11_1 = tf.get_variable("b11_1", [n_h1_en1, 1], initializer = tf.zeros_initializer())
    W11_2 = tf.get_variable("W11_2", [n_h1_en1, nx1], initializer = tf.contrib.layers.xavier_initializer())
    b11_2 = tf.get_variable("b11_2", [n_h1_en1, 1], initializer = tf.zeros_initializer())
    W21_1 = tf.get_variable("W21_1", [n_h2_en1, 2*n_h1_en1], initializer = tf.contrib.layers.xavier_initializer())
    b21_1 = tf.get_variable("b21_1", [n_h2_en1, 1], initializer = tf.zeros_initializer())
    W21_2 = tf.get_variable("W21_2", [n_h2_en1, 2*n_h1_en1], initializer = tf.contrib.layers.xavier_initializer())
    b21_2 = tf.get_variable("b21_2", [n_h2_en1, 1], initializer = tf.zeros_initializer())
    W31 = tf.get_variable("W31", [n, 2*n_h2_en1], initializer = tf.contrib.layers.xavier_initializer())
    b31 = tf.get_variable("b31", [n, 1], initializer = tf.zeros_initializer())
    # Encoder2:
    n_h1_en2 = np.floor(0.5*(nx2 + n))
    n_h2_en2 = np.floor(0.5*(n_h1_en2 + n))
    W12_1 = tf.get_variable("W12_1", [n_h1_en2, nx2], initializer = tf.contrib.layers.xavier_initializer())
    b12_1 = tf.get_variable("b12_1", [n_h1_en2, 1], initializer = tf.zeros_initializer())
    W12_2 = tf.get_variable("W12_2", [n_h1_en2, nx2], initializer = tf.contrib.layers.xavier_initializer())
    b12_2 = tf.get_variable("b12_2", [n_h1_en2, 1], initializer = tf.zeros_initializer())
    W22_1 = tf.get_variable("W22_1", [n_h2_en2, 2*n_h1_en2], initializer = tf.contrib.layers.xavier_initializer())
    b22_1 = tf.get_variable("b22_1", [n_h2_en2, 1], initializer = tf.zeros_initializer())
    W22_2 = tf.get_variable("W22_2", [n_h2_en2, 2*n_h1_en2], initializer = tf.contrib.layers.xavier_initializer())
    b22_2 = tf.get_variable("b22_2", [n_h2_en2, 1], initializer = tf.zeros_initializer())
    W32 = tf.get_variable("W32", [n, 2*n_h2_en2], initializer = tf.contrib.layers.xavier_initializer())
    b32 = tf.get_variable("b32", [n, 1], initializer = tf.zeros_initializer())
    # Decoder:
    n_h1_de = np.floor(0.5*(n + ny))
    n_h2_de = np.floor(0.5*(n_h1_de + ny))
    W4_1 = tf.get_variable("W4_1", [n_h1_de, n], initializer = tf.contrib.layers.xavier_initializer())
    b4_1 = tf.get_variable("b4_1", [n_h1_de, 1], initializer = tf.zeros_initializer())
    W4_2 = tf.get_variable("W4_2", [n_h1_de, n], initializer = tf.contrib.layers.xavier_initializer())
    b4_2 = tf.get_variable("b4_2", [n_h1_de, 1], initializer = tf.zeros_initializer())
    W5_1 = tf.get_variable("W5_1", [n_h2_de, 2*n_h1_de], initializer = tf.contrib.layers.xavier_initializer())
    b5_1 = tf.get_variable("b5_1", [n_h2_de, 1], initializer = tf.zeros_initializer())
    W5_2 = tf.get_variable("W5_2", [n_h2_de, 2*n_h1_de], initializer = tf.contrib.layers.xavier_initializer())
    b5_2 = tf.get_variable("b5_2", [n_h2_de, 1], initializer = tf.zeros_initializer())
    W6 = tf.get_variable("W6", [ny, 2*n_h2_de], initializer = tf.contrib.layers.xavier_initializer())
    b6 = tf.get_variable("b6", [ny, 1], initializer = tf.zeros_initializer())
    
    parameters =     {"W11_1":W11_1, "b11_1":b11_1, "W11_2":W11_2, "b11_2":b11_2,
                      "W21_1":W21_1, "b21_1":b21_1, "W21_2":W21_2, "b21_2":b21_2,
                      "W31":W31, "b31":b31,
                      "W12_1":W12_1, "b12_1":b12_1, "W12_2":W12_2, "b12_2":b12_2,
                      "W22_1":W22_1, "b22_1":b22_1, "W22_2":W22_2, "b22_2":b22_2,
                      "W32":W32, "b32":b32,
                      "W4_1":W4_1, "b4_1":b4_1, "W4_2":W4_2, "b4_2":b4_2,
                      "W5_1":W5_1, "b5_1":b5_1, "W5_2":W5_2, "b5_2":b5_2,
                      "W6":W6, "b6":b6}
    return parameters
"_____________________________________________________________________________"
def forward_propagation(X1, X2, parameters, avrg_pwr1, avrg_pwr2, noise_stddev, M_mb):
    #Tx1
    W11_1, b11_1 = parameters['W11_1'], parameters['b11_1']
    W11_2, b11_2 = parameters['W11_2'], parameters['b11_2']
    W21_1, b21_1 = parameters['W21_1'], parameters['b21_1']
    W21_2, b21_2 = parameters['W21_2'], parameters['b21_2']
    W31, b31 = parameters['W31'], parameters['b31']
    #Tx2
    W12_1, b12_1 = parameters['W12_1'], parameters['b12_1']
    W12_2, b12_2 = parameters['W12_2'], parameters['b12_2']
    W22_1, b22_1 = parameters['W22_1'], parameters['b22_1']
    W22_2, b22_2 = parameters['W22_2'], parameters['b22_2']
    W32, b32 = parameters['W32'], parameters['b32']
    #Decoder
    W4_1, b4_1 = parameters['W4_1'], parameters['b4_1']
    W4_2, b4_2 = parameters['W4_2'], parameters['b4_2']
    W5_1, b5_1 = parameters['W5_1'], parameters['b5_1']
    W5_2, b5_2 = parameters['W5_2'], parameters['b5_2']
    W6, b6 = parameters['W6'], parameters['b6']
    
    #Tx1
    Z11_1 = tf.matmul(W11_1,X1) + b11_1
    Z11_2 = tf.matmul(W11_2,X1) + b11_2
    A11 = tf.concat([tf.nn.relu(Z11_1),tf.nn.tanh(Z11_2)],0)
    Z21_1 = tf.matmul(W21_1,A11) + b21_1
    Z21_2 = tf.matmul(W21_2,A11) + b21_2    
    A21 = tf.concat([tf.nn.relu(Z21_1),tf.nn.tanh(Z21_2)],0)  
    Z31 = tf.matmul(W31,A21)+b31
    A31 = tf.nn.tanh(Z31) 
    
    #Tx2
    Z12_1 = tf.matmul(W12_1,X2) + b12_1
    Z12_2 = tf.matmul(W12_2,X2) + b12_2
    A12 = tf.concat([tf.nn.relu(Z12_1),tf.nn.tanh(Z12_2)],0)
    Z22_1 = tf.matmul(W22_1,A12) + b22_1
    Z22_2 = tf.matmul(W22_2,A12) + b22_2    
    A22 = tf.concat([tf.nn.relu(Z22_1),tf.nn.tanh(Z22_2)],0)  
    Z32 = tf.matmul(W32,A22)+b32
    A32 = tf.nn.tanh(Z32) 
    
    # Normalization of the channel input
    A31 = tf.div(A31, tf.sqrt( tf.reduce_mean( tf.reduce_sum(tf.multiply(A31,A31),axis=0) ))) * np.sqrt(avrg_pwr1)
    A32 = tf.div(A32, tf.sqrt( tf.reduce_mean( tf.reduce_sum(tf.multiply(A32,A32),axis=0) ))) * np.sqrt(avrg_pwr2)
    Y = A31+ A32 + tf.random_normal([A31.shape[0],M_mb],mean=0.0,stddev=noise_stddev)
    
    # Decoder
    Z4_1 = tf.matmul(W4_1,Y)+b4_1
    Z4_2 = tf.matmul(W4_2,Y)+b4_2
    A4 = tf.concat([tf.nn.relu(Z4_1),tf.nn.tanh(Z4_2)],0)
    Z5_1 = tf.matmul(W5_1,A4)+b5_1
    Z5_2 = tf.matmul(W5_2,A4)+b5_2
    A5 = tf.concat([tf.nn.relu(Z5_1),tf.nn.tanh(Z5_2)],0)
    Z6 = tf.matmul(W6,A5)+b6
    
    return A31, A32, Y, Z6
"_____________________________________________________________________________"
def compute_delivery_power(Rx, EH_Model):
    W1, b1 = EH_Model['W1'], EH_Model['b1']
    W2, b2 = EH_Model['W2'], EH_Model['b2']
    W3, b3 = EH_Model['W3'], EH_Model['b3']
    W4, b4 = EH_Model['W4'], EH_Model['b4']
    W5, b5 = EH_Model['W5'], EH_Model['b5']
    
    P_in = 4.342944819032518*tf.log(tf.reduce_sum(Rx**2,axis=0, keepdims=True))
    
    Z1 = tf.matmul(W1,P_in)+b1
    A1 = tf.nn.tanh(Z1)
    Z2 = tf.matmul(W2,A1)+b2    
    A2 = tf.nn.tanh(Z2)
    Z3 = tf.matmul(W3,A2)+b3
    A3 = tf.nn.tanh(Z3)
    Z4 = tf.matmul(W4,A3)+b4
    A4 = tf.nn.tanh(Z4)
    Z5 = tf.matmul(W5,A4)+b5
    del_power = tf.reduce_mean(tf.nn.tanh(Z5))
    
    return del_power
"_____________________________________________________________________________"
def compute_cost(Z1_lgt, Y1, Z2_lgt, Y2, Rx, lmbd, EH_Model):
    W1, b1 = EH_Model['W1'], EH_Model['b1']
    W2, b2 = EH_Model['W2'], EH_Model['b2']
    W3, b3 = EH_Model['W3'], EH_Model['b3']
    W4, b4 = EH_Model['W4'], EH_Model['b4']
    W5, b5 = EH_Model['W5'], EH_Model['b5']
    
    P_in = 4.342944819032518*tf.log(tf.reduce_sum(Rx**2,axis=0, keepdims=True))
    
    Z1 = tf.matmul(W1,P_in)+b1
    A1 = tf.nn.tanh(Z1)
    Z2 = tf.matmul(W2,A1)+b2    
    A2 = tf.nn.tanh(Z2)
    Z3 = tf.matmul(W3,A2)+b3
    A3 = tf.nn.tanh(Z3)
    Z4 = tf.matmul(W4,A3)+b4
    A4 = tf.nn.tanh(Z4)
    Z5 = tf.matmul(W5,A4)+b5
    del_power = tf.reduce_mean(tf.nn.tanh(Z5))
    
    lgts1 = tf.transpose(Z1_lgt)
    lbls1 = tf.transpose(Y1)
    lgts2 = tf.transpose(Z2_lgt)
    lbls2 = tf.transpose(Y2)
    CE1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = lgts1, labels = lbls1))
    CE2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = lgts2, labels = lbls2))
#    cost = alpha_i * CE1 + (1-alpha_i) * CE2 + lmbd/del_power
    cost = (CE1 + CE2) + lmbd/del_power
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
        
    delta_lmbd = 0.02# 0.000005
    lmbd_max = 10
    file_name = 'MAC_nx1_8_nx2_8_0p1.pickle'
    max_itr= 51
    Simulation = "Middle"
    Number_of_bits1 = 3
    Number_of_bits2 = 3
    nx1 = np.power(2,Number_of_bits1)
    nx2 = np.power(2,Number_of_bits2)
    SER = 0.98
    SER_com = 0.
    av_pwr1 = .1# change the noise as well
    av_pwr2 = .1# change the noise as well
    noise_std = 1/(np.sqrt(300.0/av_pwr1))
    n = 2
    m_tr = 10000
    m_ts = (nx1+nx2) * 50000
    mb_size = 10000 
    seed = 1                 
    num_epochs = 2201
    learning_rate = 0.01
    
    "For the start from the middle of the simulaiton"
    if Simulation == "Middle":
        lmbd = 1.011259186200933
    elif Simulation == "Start":
        lmbd = 0.
        
    "Defining the system model"
    ops.reset_default_graph()                         
    X1, X2, Y = Create_placeholders(nx1, nx2, nx1+nx2)
    M_mb = tf.placeholder(tf.int64)
    lmbd_ph = tf.placeholder(tf.float32)
    parameters = Initialize_parameters(nx1, nx2, n, nx1+nx2)
    A31, A32, Yr, Z6 = forward_propagation(X1, X2, parameters, av_pwr1, av_pwr2, noise_std, M_mb)
    cost = compute_cost(Z6[0:nx1,:], Y[0:nx1,:], Z6[nx1::,:], Y[nx1::,:], Yr, lmbd_ph, EH_Model)
    P_del_tf = compute_delivery_power(Yr, EH_Model)
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
            Constel1, Constel2, Rx, Y_est, cst, Sys_params = [], [], [], [], [], []
            cost_min = 1000.
            for i in range(max_itr):
                learning_rate = 0.01
                init = tf.global_variables_initializer()
                if i%10 == 0:
                    print("i = " + str(i) + " for nx1: "+ str(nx1) + " for nx2: "+ str(nx2) + " lambda: "+ str(lmbd))
                costs = []
                cost1=1000.
                sess.run(init)
                for epoch in range(num_epochs):
                    epoch_cost = 0.                       
                    num_minibatches = math.ceil(m_tr / mb_size) 
                    seed += 1
                    minibatches = random_mini_batches(X_train, X_train, mb_size, seed)
                    for minibatch in minibatches:
                        (minibatch_X, minibatch_Y) = minibatch
                        _ , minibatch_cost = sess.run([optimizer, cost],  feed_dict={X1: minibatch_X[0:nx1,:], X2: minibatch_X[nx1::,:], Y: minibatch_Y,
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
                    Constel1, Constel2, Rx, Y_est, cst = sess.run([A31, A32, Yr, Z6, cost],
                                                                  feed_dict={X1: X_test[0:nx1,:], X2: X_test[nx1::,:],
                                                                             Y: X_test, M_mb: X_test.shape[1], lmbd_ph: lmbd})
            SER_com = compute_SER(X_test, Y_est, nx1, nx2)
                    
            P_del = sess.run(P_del_tf, feed_dict={Yr: Rx})
            print(lmbd, SER_com, P_del, cst)
            
            plt.plot(Constel1[0,0:1000],Constel1[1,0:1000],'r.')
            plt.plot(Constel2[0,0:1000],Constel2[1,0:1000],'b.')
            plt.axis("equal")
            plt.show()
            
            if Simulation == "Middle":
                "Open the file"
                with open(file_name, 'rb') as f:
                    Optimized_params = pickle.load(f)
            elif Simulation == "Start":
                Optimized_params = []
                Simulation = "Middle"
                
            Optimized_params += [[nx1, nx2, lmbd, av_pwr1, av_pwr2, noise_std, SER_com, P_del, cst, Sys_params, Constel1[:,0:1000], Constel2[:,0:1000]]]
            lmbd += delta_lmbd+lmbd *0.05
            
            "SAVE (http://www.jessicayung.com/how-to-use-pickle-to-save-and-load-variables-in-python/)"
            with open(file_name, 'wb') as f:
                pickle.dump(Optimized_params, f)
            
    sess.close()

main()