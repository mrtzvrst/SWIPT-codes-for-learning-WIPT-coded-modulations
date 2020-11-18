import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from tensorflow.python.framework import ops

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import Fun_Lib as Fs
import scipy.io

n = 1 # complex channel use per message
k = 5 # bit per complex channel use
M = 2**k
Pa = 0.1 # avewrage power constraint
p_on = Pa/0.31 # probability of the on signal for the flash signalling
iter_length = 100
sequence_l = 500000
snr = 50
file_name = 'PP_nx_32_0p1.pickle'
Saved_file_name = 'PP_nx_32_0p1_Designed.pickle'

X_r, X_i = Fs.Generate_constellation(2**k, Pa)

Number_of_Flashes = np.argmin(np.abs(p_on-np.arange(1,M+1,1) / M))+1
Cb_r, Cb_i = X_r.reshape(-1,1), X_i.reshape(-1,1)
Code_r, Code_i = Cb_r.reshape(1,-1), Cb_i.reshape(1,-1)

"Power delivery"
"Open the EH model"
file_name_EH = 'Sys_params.pickle'
with open(file_name_EH, 'rb') as f:
    EH_Model = pickle.load(f)
Yr = tf.placeholder(tf.float32, [2,None])
P_del_tf = Fs.compute_delivery_power(Yr, EH_Model)
  
ind_sorted_symbols = np.argsort(Code_r**2+Code_i**2)[0,::-1][0:Number_of_Flashes]
Phi_t = np.zeros((Number_of_Flashes))
for i in range(Number_of_Flashes):
    t1 = ind_sorted_symbols[i]
    Phi_t[i] = np.pi*(1-np.sign(Code_r[0,t1]))/2 + np.arctan(Code_i[0,t1]/Code_r[0,t1])
t2 = np.argsort(Phi_t)

delta_vector = np.zeros((2,Number_of_Flashes))    
for i in range(Number_of_Flashes):
    t1 = ind_sorted_symbols[t2[i]]
    Phi = np.pi*(1-np.sign(Code_r[0,t1]))/2 + np.arctan(Code_i[0,t1]/Code_r[0,t1])
    radius = np.sqrt(Code_r[0,t1]**2+Code_i[0,t1]**2)
    Phi_t1 = (2*np.pi*i/Number_of_Flashes-Phi)/iter_length
    temp = (np.sqrt(M*Pa/Number_of_Flashes-1e-10)-radius)/iter_length
    delta_vector[0,t2[i]], delta_vector[1,t2[i]] = temp, Phi_t1

err_rate_vec, pdel_vec_designed = [], []

for j in range(iter_length+1):
    if j != 0:
        for i in range(Number_of_Flashes):
            
            c1, c2 = Code_r[0,ind_sorted_symbols[i]], Code_i[0,ind_sorted_symbols[i]]
            ra, th = np.sqrt(c1**2+c2**2), np.pi*(1-np.sign(c1))/2+np.arctan(c2/c1)
            
            Tz = (ra+delta_vector[0,i])*np.exp(1j*(th+delta_vector[1,i]))
            
            Code_r[0,ind_sorted_symbols[i]] = np.real(Tz)
            Code_i[0,ind_sorted_symbols[i]] = np.imag(Tz)
            
        t0 = list(set(np.arange(Code_r.shape[1]))-set(ind_sorted_symbols))
        t1 = Code_r[0,t0]
        t2 = Code_i[0,t0]
        t3 = np.sum(Code_r[0,ind_sorted_symbols]**2+Code_i[0,ind_sorted_symbols]**2)
        Code_r[0,t0] *= np.sqrt((M*Pa-np.sum(t3))/np.sum(t1**2+t2**2))
        Code_i[0,t0] *= np.sqrt((M*Pa-np.sum(t3))/np.sum(t1**2+t2**2))
#    plt.plot(Code_r[0,:],Code_i[0,:],'.')
#    plt.axis('equal')
    
    "Each roa is a codeword for an index and real and imaginary symbols ordered alternativel"
    Codebook = (np.concatenate((Code_r,Code_i),axis = 0).transpose()).reshape(-1,2*n)
#    print(np.sum(Codebook**2)/M)
    Codebook_norm = np.sum(Codebook**2,axis=1).reshape(1,-1)
    Codebook_tran = Codebook.transpose()
    
    err_rate, P_delivery = Fs.SER_Pdel_calculator(k,snr,Pa,sequence_l,Codebook,Codebook_tran,Codebook_norm,101,Yr,P_del_tf,EH_Model)
    err_rate_vec.append(err_rate)
    pdel_vec_designed.append(P_delivery)
    print(err_rate, P_delivery)
    
err_rate_vec, pdel_vec_designed = np.squeeze(err_rate_vec), np.squeeze(pdel_vec_designed)

with open(file_name, 'rb') as f:
    Optimized_params = pickle.load(f)

n = len(Optimized_params)
SER_vec = np.zeros((1,n))
Pdel_vec = np.zeros((1,n))
for i in range(n):
  SER_vec[0][i] = Optimized_params[i][4] 
  Pdel_vec[0][i] = Optimized_params[i][6] 

plt.plot(1-SER_vec[0,:],Pdel_vec[0,:],'b.',label = "NN")    
plt.plot(1-err_rate_vec,pdel_vec_designed,'r.')

plt.show()

Vec1 = []
Vec1 += [err_rate_vec]#Simulated
Vec1 += [pdel_vec_designed]
    
"Save the file"
with open(Saved_file_name, 'wb') as f:
    pickle.dump(Vec1, f)


#file_name = 'PP_nx_8_0p005.pickle'
#Saved_file_name = 'PP_nx_8_0p005.mat'
#"Save the file in MATLAB"
##Save the parameters in MATLAB
#with open(file_name, 'rb') as f:
#    evol = pickle.load(f)
#scipy.io.savemat(Saved_file_name, mdict={'evol': evol}) 
