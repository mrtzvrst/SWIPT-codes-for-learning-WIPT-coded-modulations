import pickle
import numpy as np
import matplotlib.pyplot as plt
import Fun_Lib as Fs
from scipy.special import erf
    
Dict_inf = {}
rate = 2 
Pa = 1 
sequence_l = 5000000
snr_vec_in_dB = np.arange(0,16)
snr_vec = 10**(snr_vec_in_dB/10)
fname = 'SER_vs_SNR_Rate_'+str(rate)+'.pickle'
###############################################################################
#n = 1
#k = rate*n
#"Conventional QAM Modulation"
#Codebook = Fs.QAM_gen(2**(k-1),2**(k-1),Pa=1)
#Codebook_norm = np.sum(Codebook**2,axis=1).reshape(1,-1)
#Codebook_tran = Codebook.transpose()
#
#err_vec = []
#for snr in snr_vec:
#    err_rate = Fs.SER_calculator(k,snr,1,sequence_l,Codebook,Codebook_tran,Codebook_norm,201)
#    err_vec.append(err_rate)
#    print('SNR is: '+str(snr)+', SER: '+str(err_rate))
#    
#plt.plot(snr_vec_in_dB,10*np.log10(np.asarray(err_vec)),'.-')
#plt.plot(snr_vec_in_dB,10*np.log10(1-erf(np.sqrt(snr_vec/2))),'.-')
#plt.show()

Dict_inf['Conventional_QAM'] = [snr_vec_in_dB,10*np.log10(1-erf(np.sqrt(snr_vec/2)))]
with open(fname, 'wb') as f:
    pickle.dump(Dict_inf, f)  
###############################################################################
n = 1
k = rate*n
M = 2**k * n 
"Designed Modulation"
with open('Codebook_0p005_'+str(2**k)+'_'+str(2*n)+'.pickle', 'rb') as f:
    CB = pickle.load(f)
    
Codebook_tran, Codebook_norm, Codebook = Fs.Codebook_adjusting(CB[0], CB[1], n, M, 1)

err_vec = []
for snr in snr_vec:
    err_rate = Fs.SER_calculator(k,snr,1,sequence_l,Codebook,Codebook_tran,Codebook_norm,201)
    err_vec.append(err_rate)
    print('SNR is: '+str(snr)+', SER: '+str(err_rate))
    
plt.plot(snr_vec_in_dB,10*np.log10(np.asarray(err_vec)),'.-')
plt.show()

Dict_inf['Designed_n2'] = [snr_vec_in_dB,10*np.log10(np.asarray(err_vec))]
with open(fname, 'wb') as f:
    pickle.dump(Dict_inf, f)  
###############################################################################    
n = 2
k = rate*n
M = 2**k * n 
"Designed Modulation"
with open('Codebook_0p005_'+str(2**k)+'_'+str(2*n)+'.pickle', 'rb') as f:
    CB = pickle.load(f)

Codebook_tran, Codebook_norm, Codebook = Fs.Codebook_adjusting(CB[0], CB[1], n, M, 1)

err_vec = []
for snr in snr_vec:
    err_rate = Fs.SER_calculator(k,snr,1,sequence_l,Codebook,Codebook_tran,Codebook_norm,201)
    err_vec.append(err_rate)
    print('SNR is: '+str(snr)+', SER: '+str(err_rate))
    
plt.plot(snr_vec_in_dB,10*np.log10(np.asarray(err_vec)),'.-')
plt.show()

Dict_inf['Designed_n4'] = [snr_vec_in_dB,10*np.log10(np.asarray(err_vec))]
with open(fname, 'wb') as f:
    pickle.dump(Dict_inf, f)  
###############################################################################    
n = 3
k = rate*n
M = 2**k * n 
"Designed Modulation"
with open('Codebook_0p005_'+str(2**k)+'_'+str(2*n)+'.pickle', 'rb') as f:
    CB = pickle.load(f)
    
Codebook_tran, Codebook_norm, Codebook = Fs.Codebook_adjusting(CB[0], CB[1], n, M, 1)

err_vec = []
for snr in snr_vec:
    err_rate = Fs.SER_calculator(k,snr,1,sequence_l,Codebook,Codebook_tran,Codebook_norm,201)
    err_vec.append(err_rate)
    print('SNR is: '+str(snr)+', SER: '+str(err_rate))
    
plt.plot(snr_vec_in_dB,10*np.log10(np.asarray(err_vec)),'.-')
plt.show()

Dict_inf['Designed_n6'] = [snr_vec_in_dB,10*np.log10(np.asarray(err_vec))]
with open(fname, 'wb') as f:
    pickle.dump(Dict_inf, f)  
     
"Designed Modulation"
with open(fname, 'rb') as f:
    CB = pickle.load(f)
plt.plot(CB['Conventional_QAM'][0][0:15],CB['Conventional_QAM'][1][0:15],'.-')
plt.plot(CB['Designed_n2'][0][0:15],CB['Designed_n2'][1][0:15],'.-')
plt.plot(CB['Designed_n4'][0][0:15],CB['Designed_n4'][1][0:15],'.-')
plt.plot(CB['Designed_n6'][0],CB['Designed_n6'][1],'.-')
plt.show()

#"Open the file"
#file_name = 'SER_vs_SNR_Rate_2.pickle'
#Matlab_name = 'SER_vs_SNR_Rate_2.mat'
#with open(file_name, 'rb') as f:
#    ER_vs_SNR = pickle.load(f)
#    
#"Save the file in MATLAB"
##Save the parameters in MATLAB
#with open(file_name, 'rb') as f:
#    Optimized_params = pickle.load(f)
#scipy.io.savemat(Matlab_name, mdict={'Optimized_params': Optimized_params})