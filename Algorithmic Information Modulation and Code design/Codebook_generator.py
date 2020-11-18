import pickle
#import scipy.io
import numpy as np
#import math
import matplotlib.pyplot as plt
#from scipy.stats import norm
from scipy.special import comb, perm
#import tensorflow as tf
import itertools as it
#from tensorflow.python.framework import ops
#from Functions_set import *
import Fun_Lib as Fs

"""
rate = 2, n = 2 >> dlt_ = 0.015, stp_ = 0.00001 'Small', snr = 25***
rate = 2, n = 3 >> dlt_ = 0.017, stp_ = 0.0001 'Small', snr = 25*** got error 0
rate = 2, n = 4 >> large message set file 
rate = 3, n = 2 >> dlt_ = 0.006, stp_ = 0.00001 'Small' snr = 25
rate = 3, n = 3 >> dlt_ = 0.006, stp_ = 0.00001 'Large' snr = 25***
rate = 3, n = 4 >> large message set file 
rate = 4, n = 2 >> dlt_ = 0.0025, stp_ = 0.00001 'Small' snr = 25
"""
Pa = 0.005 # This is the complex power
rate = 2 # bit per complex channel use
n = 3 # Complex channel use
Number_of_chosen_indices = 500000
dlt_ = 0.017
stp_ = 0.0001
snr = 25 
state = 'Small' # 'Large'
exist = 'Yes' # if there is code beforhand, the error probability will be compared with that 
Load_existing_permutation = 'No'
sequence_l = 1000000

k = rate*n
M = 2**k*n # 480 # 2**k*n The number of complex symbols (the sybmols of the codebook)


#Ms=n
#temp = 1
#while temp < 2000000:#2**(2*k):
#    Ms += 1
#    temp = comb(Ms,n)
#print(Ms)
#Ms = M

perm_name = 'Perm_indices'+'_'+str(2**k)+'_'+str(2*n)+'.pickle'
file_name = 'Codebook_0p'+str(Pa-int(Pa))[2:]+'_'+str(2**k)+'_'+str(2*n)+'.pickle'
X_r, X_i = Fs.Generate_constellation(M, Pa)
plt.plot(X_r,X_i,'.')

if n!= 1:
    if state == 'Large':
        if Load_existing_permutation == 'Yes':
            "Open the file"
            with open(perm_name, 'rb') as f:
                permuted_ind = pickle.load(f)
        else:
            t1 = it.permutations(np.arange(0,M),n)
            permuted_ind, s, prob = [], 0, Number_of_chosen_indices/perm(M,n)
            for i in t1:
                if np.random.binomial(1,prob)==1:
                    permuted_ind += [list(i)]
                    s+=1
                    if s % 1000==0:
                        print(s, i)
                if s==Number_of_chosen_indices:
                    break
            with open(perm_name, 'wb') as f:
                pickle.dump(permuted_ind, f)  
    else:
        t1 = it.permutations(np.arange(0,M),n)
        permuted_ind = []
        for i in t1:
            permuted_ind += [list(i)]
    permuted_ind = np.asarray(permuted_ind)    
    
    
    if exist == 'Yes':
        with open(file_name, 'rb') as f:
            Existing_Code = pickle.load(f)
            dt, err_temp = 0, Existing_Code[2]
            print('*** Previous error is: '+str(err_temp))
    else:       
        dt, err_temp = 0, 100
    
    while dt <101:    
        ind1 = np.arange(permuted_ind.shape[0])
        np.random.shuffle(ind1)
        permuted_ind = permuted_ind[ind1,:]
        X_r_rem, X_i_rem = X_r[permuted_ind], X_i[permuted_ind]
        
        d, dlt = 0, dlt_
        while d-2**k<0:
            Cb_r, Cb_i = Fs.obtain_code_given_d(X_r_rem,X_i_rem,dlt)
            dlt -= stp_
            d = len(Cb_r)
            #dlt_ = dlt + 30*stp_
        print(len(Cb_r), dlt)
            
        Cb_r, Cb_i = np.asarray(Cb_r), np.asarray(Cb_i)
        
        Cb_r, Cb_i = Cb_r[0:2**k], Cb_i[0:2**k]
        Codebook = (np.concatenate((Cb_r.reshape(1,-1),Cb_i.reshape(1,-1)),axis = 0).transpose()).reshape(-1,2*n)
        Codebook *= np.sqrt(Pa/np.sum(Codebook**2)*M)
        Codebook_norm = np.sum(Codebook**2,axis=1).reshape(1,-1)
        Codebook_tran = Codebook.transpose()
        
        # The input snr of the following funciton is the snr for real component
        err_rate = Fs.SER_calculator(k,snr,Pa,sequence_l,Codebook,Codebook_tran,Codebook_norm,11)
        
        if err_rate < err_temp:
            Code_book = [Cb_r, Cb_i, err_rate, snr]
            #plt.plot(Code_book[0],Code_book[1],'.')
            #plt.show()
            with open(file_name, 'wb') as f:
                pickle.dump(Code_book, f)  
            err_temp = err_rate
            print('*** One codebook saved: '+str(err_rate))
        
        dt += 1
        print('dt: ',str(dt),', and error rate: ',str(err_rate),', Rate: ',str(rate),', n: ',str(n))
    
else:
    Cb_r, Cb_i = X_r.reshape(-1,1), X_i.reshape(-1,1)
    Code_book = [Cb_r, Cb_i]
    #plt.plot(Cb_r,Cb_i,'.')
    with open(file_name, 'wb') as f:
        pickle.dump(Code_book, f)  
   

#with open('Codebook_0p005_64_6.pickle', 'rb') as f:
#    Codebook = pickle.load(f)
#print(Codebook[2])