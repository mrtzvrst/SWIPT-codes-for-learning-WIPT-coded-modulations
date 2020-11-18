import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf
import numpy as np
import math 
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
import pickle 

   
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


def Create_placeholders(n_x, n_y):
    X = tf.placeholder(tf.float32, [n_x,None])
    Y = tf.placeholder(tf.float32, [n_y,None])
    return X, Y
"_____________________________________________________________________________"
def Initialize_parameters(n_x, n, n_y): # n is the number of channel use
    # Encoder:
    n_h1_en = np.floor(0.6*(n_x + n))
    n_h2_en = np.floor(0.6*(n_h1_en + n))
    W1_1 = tf.get_variable("W1_1", [n_h1_en, n_x], initializer = tf.contrib.layers.xavier_initializer())
    b1_1 = tf.get_variable("b1_1", [n_h1_en, 1], initializer = tf.zeros_initializer())
    W1_2 = tf.get_variable("W1_2", [n_h1_en, n_x], initializer = tf.contrib.layers.xavier_initializer())
    b1_2 = tf.get_variable("b1_2", [n_h1_en, 1], initializer = tf.zeros_initializer())
    W2_1 = tf.get_variable("W2_1", [n_h2_en, 2*n_h1_en], initializer = tf.contrib.layers.xavier_initializer())
    b2_1 = tf.get_variable("b2_1", [n_h2_en, 1], initializer = tf.zeros_initializer())
    W2_2 = tf.get_variable("W2_2", [n_h2_en, 2*n_h1_en], initializer = tf.contrib.layers.xavier_initializer())
    b2_2 = tf.get_variable("b2_2", [n_h2_en, 1], initializer = tf.zeros_initializer())
    W3 = tf.get_variable("W3", [n, 2*n_h2_en], initializer = tf.contrib.layers.xavier_initializer())
    b3 = tf.get_variable("b3", [n, 1], initializer = tf.zeros_initializer())
    
    # Decoder:
    n_h1_de = np.floor(0.6*(n + n_x))
    n_h2_de = np.floor(0.6*(n_h1_de + n_x))
    W4_1 = tf.get_variable("W4_1", [n_h1_de, n], initializer = tf.contrib.layers.xavier_initializer())
    b4_1 = tf.get_variable("b4_1", [n_h1_de, 1], initializer = tf.zeros_initializer())
    W4_2 = tf.get_variable("W4_2", [n_h1_de, n], initializer = tf.contrib.layers.xavier_initializer())
    b4_2 = tf.get_variable("b4_2", [n_h1_de, 1], initializer = tf.zeros_initializer())
    W5_1 = tf.get_variable("W5_1", [n_h2_de, 2*n_h1_de], initializer = tf.contrib.layers.xavier_initializer())
    b5_1 = tf.get_variable("b5_1", [n_h2_de, 1], initializer = tf.zeros_initializer())
    W5_2 = tf.get_variable("W5_2", [n_h2_de, 2*n_h1_de], initializer = tf.contrib.layers.xavier_initializer())
    b5_2 = tf.get_variable("b5_2", [n_h2_de, 1], initializer = tf.zeros_initializer())
    W6 = tf.get_variable("W6", [n_y, 2*n_h2_de], initializer = tf.contrib.layers.xavier_initializer())
    b6 = tf.get_variable("b6", [n_y, 1], initializer = tf.zeros_initializer())
    
    parameters = {"W1_1":W1_1, "b1_1":b1_1, "W1_2":W1_2, "b1_2":b1_2,
                  "W2_1":W2_1, "b2_1":b2_1, "W2_2":W2_2, "b2_2":b2_2,
                  "W3":W3, "b3":b3, 
                  "W4_1":W4_1, "b4_1":b4_1, "W4_2":W4_2, "b4_2":b4_2,
                  "W5_1":W5_1, "b5_1":b5_1, "W5_2":W5_2, "b5_2":b5_2,
                  "W6":W6, "b6":b6}
    return parameters
"_____________________________________________________________________________"
def forward_propagation(X, parameters, avrg_pwr, noise_stddev, M_mb):
    W1_1, b1_1 = parameters['W1_1'], parameters['b1_1']
    W1_2, b1_2 = parameters['W1_2'], parameters['b1_2']
    W2_1, b2_1 = parameters['W2_1'], parameters['b2_1']
    W2_2, b2_2 = parameters['W2_2'], parameters['b2_2']
    W3, b3 = parameters['W3'], parameters['b3']

    W4_1, b4_1 = parameters['W4_1'], parameters['b4_1']
    W4_2, b4_2 = parameters['W4_2'], parameters['b4_2']
    W5_1, b5_1 = parameters['W5_1'], parameters['b5_1']
    W5_2, b5_2 = parameters['W5_2'], parameters['b5_2']
    W6, b6 = parameters['W6'], parameters['b6']
    
    #Z1 = tf.nn.batch_normalization(tf.matmul(W1,X)+b1, mean=0, variance=1, offset=0, scale=1, variance_epsilon= 1e-8)
    Z1_1 = tf.matmul(W1_1,X) + b1_1
    Z1_2 = tf.matmul(W1_2,X) + b1_2
    A1 = tf.concat([tf.nn.relu(Z1_1),tf.nn.tanh(Z1_2)],0)
#    A1 = tf.nn.dropout(A1,0.95)
    #Z2 = tf.nn.batch_normalization(tf.matmul(W2,A1)+b2, mean=0, variance=1, offset=0, scale=1, variance_epsilon= 1e-8)
    Z2_1 = tf.matmul(W2_1,A1) + b2_1
    Z2_2 = tf.matmul(W2_2,A1) + b2_2    
    A2 = tf.concat([tf.nn.relu(Z2_1),tf.nn.tanh(Z2_2)],0)  
#    A2 = tf.nn.dropout(A2,0.95)
    Z3 = tf.matmul(W3,A2)+b3
    A3 = tf.nn.tanh(Z3)
    ## Normalization of the channel input
    A3 = tf.div(A3, tf.sqrt( tf.reduce_mean( tf.reduce_mean(tf.multiply(A3,A3),axis=0) )*2)) * np.sqrt(avrg_pwr)
    Y = A3 + tf.random_normal([A3.shape[0],M_mb],mean=0.0,stddev=noise_stddev)
    
    #Z4 = tf.nn.batch_normalization(tf.matmul(W4,Y)+b4, mean=0, variance=1, offset=0, scale=1, variance_epsilon= 1e-8)
    Z4_1 = tf.matmul(W4_1,Y)+b4_1
    Z4_2 = tf.matmul(W4_2,Y)+b4_2
    A4 = tf.concat([tf.nn.relu(Z4_1),tf.nn.tanh(Z4_2)],0)
    #Z5 = tf.nn.batch_normalization(tf.matmul(W5,A4)+b5, mean=0, variance=1, offset=0, scale=1, variance_epsilon= 1e-8)
    Z5_1 = tf.matmul(W5_1,A4)+b5_1
    Z5_2 = tf.matmul(W5_2,A4)+b5_2
    A5 = tf.concat([tf.nn.relu(Z5_1),tf.nn.tanh(Z5_2)],0)
    Z6 = tf.matmul(W6,A5)+b6
    return A3, Y, Z6
"_____________________________________________________________________________"
def compute_delivery_power(Rx, EH_Model):
    W1, b1 = EH_Model['W1'], EH_Model['b1']
    W2, b2 = EH_Model['W2'], EH_Model['b2']
    W3, b3 = EH_Model['W3'], EH_Model['b3']
    W4, b4 = EH_Model['W4'], EH_Model['b4']
    W5, b5 = EH_Model['W5'], EH_Model['b5']
    
    Rx_new = tf.transpose(tf.reshape(tf.transpose(Rx),[-1,2]))
    P_in = 4.342944819032518*tf.log(tf.reduce_sum(Rx_new**2,axis=0, keepdims=True))
    
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
def compute_cost(Z, Y, Rx, lmbd, EH_Model):
    W1, b1 = EH_Model['W1'], EH_Model['b1']
    W2, b2 = EH_Model['W2'], EH_Model['b2']
    W3, b3 = EH_Model['W3'], EH_Model['b3']
    W4, b4 = EH_Model['W4'], EH_Model['b4']
    W5, b5 = EH_Model['W5'], EH_Model['b5']
    
#    b = np.array([np.arange(1,38,4),np.arange(2,39,4),np.arange(3,40,4),np.arange(4,41,4)])
#    b = b.transpose()
#    b.reshape(-1,2).transpose()

    Rx_new = tf.transpose(tf.reshape(tf.transpose(Rx),[-1,2]))
    P_in = 4.342944819032518*tf.log(tf.reduce_sum(Rx_new**2,axis=0, keepdims=True))
    
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
    
    lgts = tf.transpose(Z)
    lbls = tf.transpose(Y)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = lgts, labels = lbls)) + lmbd/del_power
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
def compute_SER(Y, Y_est):
    SER = np.sum((Y_est.argmax(axis=0)-Y.argmax(axis=0))!=0)/(Y.shape[1])
    return SER 
"_____________________________________________________________________________"
def main():
#    config = tf. ConfigProto()
#    config.gpu_options.per_process_gpu_memory_fraction = 0.02
    
#    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.0001)
#    conf = tf.ConfigProto(gpu_options = gpu_options)
#    trainngConfig = tf.estimator.RunConfig(session_config=conf)

    "Open the EH model"
    file_name = 'Sys_params.pickle'
    with open(file_name, 'rb') as f:
        EH_Model = pickle.load(f)
        
    "Parameterization"
    n = 6
    delta_lmbd = 0.000002#0.00005,0.001
    lmbd_max = 2
    file_name = 'pp_nx_64_0p005_L6.pickle'
    max_itr= 51
    Simulation = "Middle"
    nx = 6
    av_pwr =0.005 #0.1 
    
    ny = 2**nx
    SER = 0.99
    SER_com = 0.
    
    noise_std = 1/(np.sqrt(100./av_pwr))
    m_tr = 8000
    m_ts = nx * 10000
    mb_size = 8000
    seed = 1                 
    num_epochs = 8000
    learning_rate = 0.005
    
    "For the start from the middle of the simulaiton"
    if Simulation == "Middle":
        lmbd = 0.00010405447522176576
    elif Simulation == "Start":
        lmbd = 0.
    
    "Defining the system model"
    ops.reset_default_graph()  
    X, Y = Create_placeholders(nx,ny)
    M_mb = tf.placeholder(tf.int64)
    lmbd_ph = tf.placeholder(tf.float32)
    parameters = Initialize_parameters(nx,n,ny)
    A3, Yr, Z6 = forward_propagation(X, parameters, av_pwr, noise_std, M_mb)
    cost = compute_cost(Z6, Y, Yr, lmbd_ph, EH_Model)
    P_del_tf = compute_delivery_power(Yr, EH_Model)
    optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08, name='Adam').minimize(cost)
   
    "Producing the training and test datas"
    X_test_ind = np.squeeze(np.random.randint(ny,size=[1,m_ts]))
    X_train_ind = np.squeeze(np.random.randint(ny,size=[1,m_tr]))
    X_test = (((X_test_ind[:,None] & (1 << np.arange(nx)))) > 0).astype(float).transpose()
    X_train = (((X_train_ind[:,None] & (1 << np.arange(nx)))) > 0).astype(float).transpose()
    X_test_oh = tf.one_hot(X_test_ind, depth = ny, axis = 0)
    X_train_oh = tf.one_hot(X_train_ind, depth = ny, axis = 0)
      
    "Optimization"
    with tf.Session() as sess:
        Y_test = sess.run(X_test_oh)
        Y_train = sess.run(X_train_oh)
        while (SER_com <= SER) and lmbd < lmbd_max: 
            Constel, Rx, Y_est, cst, Sys_params = [], [], [], [], []
            cost_min = 10000000000000.
            for i in range(max_itr):
                learning_rate = 0.01
                init = tf.global_variables_initializer()
                if i%20 == 0:
                    print("i = " + str(i) + " for nx: "+ str(nx) + ", n: " + str(n) + " lambda: "+ str(lmbd))
                costs = []
                cost1=1000.
                sess.run(init)
                for epoch in range(num_epochs):
                    epoch_cost = 0.                       
                    num_minibatches = math.ceil(m_tr / mb_size) 
                    seed += 1
                    minibatches = random_mini_batches(X_train, Y_train, mb_size, seed)
                    for minibatch in minibatches:
                        (minibatch_X, minibatch_Y) = minibatch
                        _ , minibatch_cost = sess.run([optimizer, cost],  feed_dict={X: minibatch_X, Y: minibatch_Y,
                                                                                     M_mb: minibatch_X.shape[1], lmbd_ph: lmbd})
                        
                        epoch_cost += minibatch_cost / num_minibatches
                        
#                    if epoch%10 == 0:
#                        print(epoch_cost)
                    costs.append(epoch_cost)
                    if epoch!= 0 and epoch % 60 == 0:
                        cost2 = np.mean(costs[-50::])
                        if cost2 < cost1 :
                            cost1 = np.copy(cost2)
                        else:
                            learning_rate/=1.1 
                            # print(learning_rate)
        
                    if learning_rate<0.000001 or np.isnan(epoch_cost):
                        print('Optimization break because of small learning rate or cost = nan')
                        break
#                plt.plot(costs)
#                plt.axis([0, num_epochs, 0, 1])
#                plt.show()
#                
                if epoch_cost < cost_min and np.isnan(epoch_cost)!=1:
                    cost_min = epoch_cost
                    Sys_params = sess.run(parameters)
                    "Obtaining the constellaitons and the cost"
                    Constel, Rx, Y_est, cst = sess.run([A3, Yr, Z6, cost], 
                                                       feed_dict={X: X_test, Y: Y_test, M_mb: X_test.shape[1], lmbd_ph: lmbd})
                 
            SER_com = compute_SER(Y_test, Y_est)
                    
            P_del = sess.run(P_del_tf, feed_dict={Yr: Rx})
            print(nx, n, av_pwr, lmbd, SER_com, P_del, cst)
            
            plt.plot(Constel[0,0:1000],Constel[1,0:1000],'r.')
#            plt.plot(Constel[2,0:1000],Constel[3,0:1000],'b.')
            plt.axis("equal")
            plt.show()
            
            if Simulation == "Middle":
                "Open the file"
                with open(file_name, 'rb') as f:
                    Optimized_params = pickle.load(f)
            elif Simulation == "Start":
                Optimized_params = []
                Simulation = "Middle"
                
            Optimized_params += [[nx, lmbd, av_pwr, noise_std, SER_com, cst, P_del, Sys_params, Constel[:,0:500]]]
            
            
            "SAVE (http://www.jessicayung.com/how-to-use-pickle-to-save-and-load-variables-in-python/)"
            with open(file_name, 'wb') as f:
                pickle.dump(Optimized_params, f)
            
            lmbd += delta_lmbd+lmbd*0.01
            
    sess.close()
main()


