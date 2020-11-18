import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import Fun_Lib as Fs
import scipy.io



n = 4 # complex channel use per message
rate = 2 # bit per complex channel use
Pa = 0.005 # avewrage power constraint
iter_length = 50


k = rate*n #number of bits per complex symbol
M = 2**k * n # total 
Code_book_name = 'Codebook_0p'+str(Pa-int(Pa))[2:]+'_'+str(2**k)+'_'+str(2*n)+'.pickle'
p_on = Pa/0.316  # probability of the on signal for the flash signalling
  
"Codebook"
with open(Code_book_name, 'rb') as f:
    CB = pickle.load(f)
Number_of_Flashes = np.argmin(np.abs(p_on-np.arange(1,M+1,1) / M))+1
Cb_r, Cb_i = CB[0], CB[1]
Code_r, Code_i = Cb_r.reshape(1,-1), Cb_i.reshape(1,-1)
   
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
    #plt.plot(Code_r[0,:],Code_i[0,:],'.')
    #plt.axis('equal')
    params += [Codebook]
 

evolution_r = np.zeros((2**k,n,len(params)))
evolution_i = np.zeros((2**k,n,len(params)))
for j in range(2**k):
    for i in range(len(params)):
        for l in range(n):
            evolution_r[j,l,i]=params[i][j,2*l]
            evolution_i[j,l,i]=params[i][j,2*l+1]

for i in range(2**k):
    for j in range(n):
        plt.plot(evolution_r[i,j,:],evolution_i[i,j,:],'-k')
        plt.plot(evolution_r[i,j,0],evolution_i[i,j,0],'ok')
        plt.plot(evolution_r[i,j,-1],evolution_i[i,j,-1],'*k')

plt.axis('equal')
plt.show()

evol = [evolution_r,evolution_i]
scipy.io.savemat('n_length_Modulation_Evolution_0p005.mat', mdict={'evol': evol}) 



