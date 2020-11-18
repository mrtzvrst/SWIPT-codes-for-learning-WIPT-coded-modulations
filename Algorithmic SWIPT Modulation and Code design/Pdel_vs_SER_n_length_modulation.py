import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import Fun_Lib as Fs
import scipy.io


def main():
    n = 3 # complex channel use per message
    rate = 3 # bit per complex channel use
    Pa = 0.005 # avewrage power constraint
    iter_length = 100
    sequence_l = 1000000
    snr = 50 #dB
    
    k = rate*n #number of bits per complex symbol
    file_name = 'analytic_code_pp_nx_'+str(2**k)+'_0p005_L'+str(2*n)+'.pickle'
    M = 2**k * n # total 
    Code_book_name = 'Codebook_0p'+str(Pa-int(Pa))[2:]+'_'+str(2**k)+'_'+str(2*n)+'.pickle'
    p_on = Pa/0.31  # probability of the on signal for the flash signalling
    
  
    "Power delivery"
    "Open the EH model"
    file_name_EH = 'Sys_params.pickle'
    with open(file_name_EH, 'rb') as f:
        EH_Model = pickle.load(f)
        
    "Codebook"
    with open(Code_book_name, 'rb') as f:
        CB = pickle.load(f)
    Number_of_Flashes = np.argmin(np.abs(p_on-np.arange(1,M+1,1) / M))+1
    Cb_r, Cb_i = CB[0], CB[1]
    Code_r, Code_i = Cb_r.reshape(1,-1), Cb_i.reshape(1,-1)
   
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

    params = []
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
            
        "Each row is a codeword for a message index and real and imaginary symbols ordered alternativel"
        Codebook = (np.concatenate((Code_r,Code_i),axis = 0).transpose()).reshape(-1,2*n)
        # print(np.sum(Codebook**2)/M)
        Codebook_norm = np.sum(Codebook**2,axis=1).reshape(1,-1)
        Codebook_tran = Codebook.transpose()
        
        #plt.plot(Code_r,Code_i,'.')
        
        err_rate, P_delivery = Fs.SER_Pdel_calculator(k,snr,Pa,sequence_l,Codebook,Codebook_tran,Codebook_norm,101,Yr,P_del_tf,EH_Model)
        print(err_rate, P_delivery)
        
        params += [[n, k, p_on, Pa, snr, err_rate, P_delivery, Codebook]]
        
        "Save the file"
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)
main()


"""
"Open the EH model"
file_name_EH = 'analytic_code_pp_nx_4_0p005_L2.pickle'
file_name_mat = 'analytic_code_pp_nx_4_0p005_L2.mat'
with open(file_name_EH, 'rb') as f:
    params = pickle.load(f)
n = len(params)
SER_vec = np.zeros((n))
Pdel_vec = np.zeros((n))
for i in range(n):
    SER_vec[i] = 1-params[i][5]
    Pdel_vec[i] = params[i][6]
plt.plot(SER_vec,Pdel_vec,'.-')
scipy.io.savemat(file_name_mat, mdict={'params': params}) 

file_name_EH = 'analytic_code_pp_nx_16_0p005_L4.pickle'
file_name_mat = 'analytic_code_pp_nx_16_0p005_L4.mat'
with open(file_name_EH, 'rb') as f:
    params = pickle.load(f)
n = len(params)
SER_vec = np.zeros((n))
Pdel_vec = np.zeros((n))
for i in range(n):
    SER_vec[i] = 1-params[i][5]
    Pdel_vec[i] = params[i][6]
plt.plot(SER_vec,Pdel_vec,'.-')
scipy.io.savemat(file_name_mat, mdict={'params': params}) 

file_name_EH = 'analytic_code_pp_nx_64_0p005_L6.pickle'
file_name_mat = 'analytic_code_pp_nx_64_0p005_L6.mat'
with open(file_name_EH, 'rb') as f:
    params = pickle.load(f)
n = len(params)
SER_vec = np.zeros((n))
Pdel_vec = np.zeros((n))
for i in range(n):
    SER_vec[i] = 1-params[i][5]
    Pdel_vec[i] = params[i][6]
plt.plot(SER_vec,Pdel_vec,'.-')
#plt.axis([0.9999999,1.000001,0.0004,0.0008])
scipy.io.savemat(file_name_mat, mdict={'params': params}) 

file_name_EH = 'analytic_code_pp_nx_256_0p005_L8.pickle'
file_name_mat = 'analytic_code_pp_nx_256_0p005_L8.mat'
with open(file_name_EH, 'rb') as f:
    params = pickle.load(f)
n = len(params)
SER_vec = np.zeros((n))
Pdel_vec = np.zeros((n))
for i in range(n):
    SER_vec[i] = 1-params[i][5]
    Pdel_vec[i] = params[i][6]
plt.plot(SER_vec,Pdel_vec,'.-')
#plt.axis([0.9999999,1.000001,0.0004,0.0008])
scipy.io.savemat(file_name_mat, mdict={'params': params}) 
plt.show()

###############################

file_name_EH = 'analytic_code_pp_nx_8_0p005_L2.pickle'
file_name_mat = 'analytic_code_pp_nx_8_0p005_L2.mat'
with open(file_name_EH, 'rb') as f:
    params = pickle.load(f)
n = len(params)
SER_vec = np.zeros((n))
Pdel_vec = np.zeros((n))
for i in range(n):
    SER_vec[i] = 1-params[i][5]
    Pdel_vec[i] = params[i][6]
plt.plot(SER_vec,Pdel_vec,'.-b')
scipy.io.savemat(file_name_mat, mdict={'params': params}) 

file_name_EH = 'analytic_code_pp_nx_64_0p005_L4.pickle'
file_name_mat = 'analytic_code_pp_nx_64_0p005_L4.mat'
with open(file_name_EH, 'rb') as f:
    params = pickle.load(f)
n = len(params)
SER_vec = np.zeros((n))
Pdel_vec = np.zeros((n))
for i in range(n):
    SER_vec[i] = 1-params[i][5]
    Pdel_vec[i] = params[i][6]
plt.plot(SER_vec,Pdel_vec,'.-k')
scipy.io.savemat(file_name_mat, mdict={'params': params}) 

file_name_EH = 'analytic_code_pp_nx_512_0p005_L6.pickle'
file_name_mat = 'analytic_code_pp_nx_512_0p005_L6.mat'
with open(file_name_EH, 'rb') as f:
    params = pickle.load(f)
n = len(params)
SER_vec = np.zeros((n))
Pdel_vec = np.zeros((n))
for i in range(n):
    SER_vec[i] = 1-params[i][5]
    Pdel_vec[i] = params[i][6]
plt.plot(SER_vec,Pdel_vec,'.-')
scipy.io.savemat(file_name_mat, mdict={'params': params}) 
"""