import pickle
import scipy.io
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.special import comb, perm
import tensorflow as tf
import itertools as it
from tensorflow.python.framework import ops


def gen_points_phase_table():
    N_circle = 10000
    point_circles = np.zeros((N_circle+1,), dtype = int)
    phase_circles = np.zeros((N_circle+1,), dtype = int)
    point_circles[0] = 1
    point_circles[1] = 6
    for m in range(2,N_circle+1):
        point_circles[m] = math.floor((np.pi/np.arcsin(1/(2*(m)))))
    return point_circles, phase_circles
"_____________________________________________________________________________"

def gen_constellation(M, point_circles, phase_circles,P_tx):
     sum_ = 0  
     for i in range(len(point_circles)):
         sum_ += point_circles[i]
         if sum_ >= M:
             break
     N_last_circle = point_circles[i] +M - sum_
     idx_last_circle = i 
     Re_ = np.zeros(1)
     Im_ = np.zeros(1)
     r_= 1
     norm_pow = 0
     beg_last_circle = 1
     for i in range(1,idx_last_circle):
         R_ = i*r_
         shift = 2*np.pi/point_circles[i]
         norm_pow += point_circles[i]*np.power(i,2)
         for j in range(point_circles[i]):
             Re_ = np.append(Re_, R_*np.cos(shift*j+phase_circles[i]))
             Im_ = np.append(Im_,R_*np.sin(shift*j+phase_circles[i]))
             beg_last_circle +=1
     R_ = idx_last_circle*r_
     shift = 2*np.pi/N_last_circle
     norm_pow_not_last = norm_pow
     norm_pow += N_last_circle*np.power(idx_last_circle,2)
     for j in range(N_last_circle):
         Re_ = np.append(Re_,R_*np.cos(shift*j+phase_circles[idx_last_circle]))
         Im_ = np.append(Im_,R_*np.sin(shift*j+phase_circles[idx_last_circle])) 
     r = np.sqrt(P_tx*M/norm_pow)
     for i in range(M):
         Re_[i] *=r
         Im_[i] *=r   
     return Re_, Im_, r, beg_last_circle, idx_last_circle, norm_pow, norm_pow_not_last, N_last_circle
"_____________________________________________________________________________"

def Generate_constellation(M, Ptx):
    point_circles, phase_circles = gen_points_phase_table()
    Re_it,Im_it,r,_,_,_,_,_ = gen_constellation(M, point_circles, phase_circles,Ptx)
#    plt.plot(Re_it,Im_it,'r.')
#    plt.xlabel('Real')
#    plt.ylabel('Imaginary')
#    plt.axis('equal')
#    plt.show()
    return Re_it, Im_it
"_____________________________________________________________________________"

def obtain_code_given_d(X_r_rem,X_i_rem,dlt):
    Cb_r, Cb_i = [], []
    flag = 0
    while flag==0:
        
        if X_r_rem.shape[0] > 1:
            t1, t2 = X_r_rem[0,:], X_i_rem[0,:]
            Cb_r += [t1]
            Cb_i += [t2]
            X_r_rem, X_i_rem = X_r_rem[1::,:],  X_i_rem[1::,:]
            dist_vec = np.sum((t1 - X_r_rem)**2 + (t2 - X_i_rem)**2,axis=1)
            sorted_dis = np.argsort(dist_vec)
            sorted_dis = sorted_dis[dist_vec[sorted_dis]>dlt]
            if sorted_dis.shape[0]==0:
                flag = 1
            else:
                X_r_rem, X_i_rem = X_r_rem[sorted_dis], X_i_rem[sorted_dis] 
        else:
            flag = 1
    return Cb_r, Cb_i
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

def QAM_gen(nx,ny,Pa=1):
    Nx, Ny = np.linspace(-1,1,nx), np.linspace(-1,1,ny)
    cart_prod = np.array([x for x in it.product(Nx, Ny)])
    cart_prod *= np.sqrt(Pa/np.sum(cart_prod**2)*nx*ny)
    return cart_prod
"_____________________________________________________________________________"
# the complex noise variance is Pa/snr which is the variance of the real component
def SER_calculator(k,snr,Pa,sequence_l,Codebook,Codebook_tran,Codebook_norm,itr):
    temp = 0
    err_rate = 0
    
    while temp < itr:
        X_test_ind = np.squeeze(np.random.randint(2**k,size=[1,sequence_l]))
        Tx = Codebook[X_test_ind,:]
        Rx = Tx + np.random.normal(0 , 1/(np.sqrt(snr/Pa*2)), size=(Tx.shape))
        esti = np.argmin(Codebook_norm - 2*np.matmul(Rx,Codebook_tran),axis=1)
        err_rate += 1-np.sum(X_test_ind==esti)/sequence_l
        temp += 1
    err_rate /= temp
    return err_rate
"_____________________________________________________________________________"
# the complex noise variance is Pa/snr which is the variance of the real component
def SER_Pdel_calculator(k,snr,Pa,sequence_l,Codebook,Codebook_tran,Codebook_norm,itr,Yr,P_del_tf,EH_Model):
    temp = 0
    err_rate, P_delivery = 0, 0
    
    while temp < itr:
        X_test_ind = np.squeeze(np.random.randint(2**k,size=[1,sequence_l]))
        Tx = Codebook[X_test_ind,:]
        Rx = Tx + np.random.normal(0 , 1/(np.sqrt(snr/Pa*2)), size=(Tx.shape))
        esti = np.argmin(Codebook_norm - 2*np.matmul(Rx,Codebook_tran),axis=1)
        err_rate += 1-np.sum(X_test_ind==esti)/sequence_l
        
        Rx_resh = Rx.reshape(-1,2).transpose()
        with tf.Session() as sess:
            P_del = sess.run(P_del_tf, feed_dict={Yr: Rx_resh})
        P_delivery += P_del
        
        temp += 1
    P_delivery /= temp
    err_rate /= temp
    return err_rate, P_delivery
"_____________________________________________________________________________"

def Codebook_adjusting(CB0, CB1, n, M, Pa):
    Cb_r, Cb_i = CB0, CB1
    Code_r, Code_i = Cb_r.reshape(1,-1), Cb_i.reshape(1,-1)
    Codebook = (np.concatenate((Code_r,Code_i),axis = 0).transpose()).reshape(-1,2*n)
    Codebook *= np.sqrt(1/np.sum(Codebook**2)*M)
    Codebook_norm = np.sum(Codebook**2,axis=1).reshape(1,-1)
    Codebook_tran = Codebook.transpose()
    return Codebook_tran, Codebook_norm, Codebook






