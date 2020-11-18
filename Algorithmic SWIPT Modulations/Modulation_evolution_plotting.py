import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import Fun_Lib as Fs
import scipy.io

n = 1 # complex channel use per message
k = 5 # bit per complex channel use
M = 2**k
Pa = 0.12 # avewrage power constraint
p_on = Pa/0.317 # probability of the on signal for the flash signalling
iter_length = 50

X_r, X_i = Fs.Generate_constellation(2**k, Pa)
'Permutation Coding'
Number_of_Flashes = np.argmin(np.abs(p_on-np.arange(1,M+1,1) / M))+1
Cb_r, Cb_i = X_r.reshape(-1,1), X_i.reshape(-1,1)
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
    
    "Each roa is a codeword for an index and real and imaginary symbols ordered alternativel"
    Codebook = (np.concatenate((Code_r,Code_i),axis = 0).transpose()).reshape(-1,2*n)
    Codebook *= np.sqrt(Pa/np.sum(Codebook**2)*M)
    
    params += [Codebook]
    
    plt.plot(Code_r[0,:],Code_i[0,:],'.')
    plt.axis('equal')
plt.show()

evolution_r = np.zeros((2**k,len(params)))
evolution_i = np.zeros((2**k,len(params)))
for j in range(2**k):
    for i in range(len(params)):
        evolution_r[j,i]=params[i][j,0]
        evolution_i[j,i]=params[i][j,1]

for i in range(2**k):
    plt.plot(evolution_r[i,:],evolution_i[i,:],'-k')
    plt.plot(evolution_r[i,0],evolution_i[i,0],'ok')
    plt.plot(evolution_r[i,-1],evolution_i[i,-1],'*k')

plt.axis('equal')
plt.show()

#evol = [evolution_r,evolution_i]
#"Save the file"
#with open('Modulation_Evolution_0p1.pickle', 'wb') as f:
#    pickle.dump(evol, f)
#
#"Save the file in MATLAB"
##Save the parameters in MATLAB
#with open('Modulation_Evolution_0p1.pickle', 'rb') as f:
#    evol = pickle.load(f)
#scipy.io.savemat('Modulation_Evolution_0p1.mat', mdict={'evol': evol}) 
