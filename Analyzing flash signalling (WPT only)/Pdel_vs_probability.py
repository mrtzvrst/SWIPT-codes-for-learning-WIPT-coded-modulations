import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle 
import scipy.io

def compute_delivery_power(Rx, EH_Model):
    W1, b1 = EH_Model['W1'], EH_Model['b1']
    W2, b2 = EH_Model['W2'], EH_Model['b2']
    W3, b3 = EH_Model['W3'], EH_Model['b3']
    W4, b4 = EH_Model['W4'], EH_Model['b4']
    W5, b5 = EH_Model['W5'], EH_Model['b5']
    P_in = 4.342944819032518*tf.log(tf.reduce_sum(Rx,axis=0, keepdims=True))
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

file_name = 'Sys_params.pickle'
with open(file_name, 'rb') as f:
    EH_Model = pickle.load(f)
    
m_tr = 1000000
d = 0.001
Pa_vec = np.array([0.005, 0.05, 0.2, 0.4])

Rx = tf.placeholder(tf.float32, [1,None])
Pdel = compute_delivery_power(Rx, EH_Model)
Prob_vec = np.arange(0.001,1,d)

parameters = []
for Pa in Pa_vec:
    Pd = []
    with tf.Session() as sess:
        for i in Prob_vec:
            X = np.random.binomial(1,i,m_tr).reshape(1,m_tr)
            X = X.astype(float)
            X_normalized = X*np.sqrt(Pa/np.mean(X))#+np.random.normal(0,1/(np.sqrt(100./Pa)),(1,m_tr))+1j*np.random.normal(0,1/(np.sqrt(100./Pa)),(1,m_tr))
            X_power = np.abs(X_normalized)**2
            Pd.append(sess.run(Pdel, feed_dict={Rx:X_power}))
    sess.close()
    x = np.array(Pd)
    plt.plot(Prob_vec,np.array(Pd)/np.max(x[~np.isnan(np.array(x))]))
    parameters += [[Prob_vec, np.array(Pd)/np.max(x[~np.isnan(np.array(x))])]]
plt.show()
with open('Prob_vs_Power.pickle', 'wb') as f:
    pickle.dump(parameters, f)


Pa_vec = np.arange(0.001,0.4,0.001)
Optimal_Prob = []
for Pa in Pa_vec:
    print(Pa)
    Pd = []
    with tf.Session() as sess:
        for i in Prob_vec:
            X = np.random.binomial(1,i,m_tr).reshape(1,m_tr)
            X = X.astype(float)
            X_normalized = X/np.mean(X)*Pa
            Pd.append(sess.run(Pdel, feed_dict={Rx:X_normalized}))
    x = np.array(Pd)
    Optimal_Prob.append(Prob_vec[np.argmax(x[~np.isnan(np.array(x))])])
plt.plot(Pa_vec,np.array(Optimal_Prob),'.')
parameters = [[Pa_vec,np.array(Optimal_Prob)]]
with open('Optimal_Prob_vs_Power.pickle', 'wb') as f:
    pickle.dump(parameters, f)


"Open the file"
file_name = 'Optimal_Prob_vs_Power.pickle'
Matlab_name = 'Optimal_Prob_vs_Power.mat'
with open(file_name, 'rb') as f:
    Optimized_params = pickle.load(f)
scipy.io.savemat(Matlab_name, mdict={'Optimized_params': Optimized_params})

"Open the file"
file_name = 'Prob_vs_Power.pickle'
Matlab_name = 'Prob_vs_Power.mat'
with open(file_name, 'rb') as f:
    Optimized_params = pickle.load(f)
scipy.io.savemat(Matlab_name, mdict={'Optimized_params': Optimized_params})