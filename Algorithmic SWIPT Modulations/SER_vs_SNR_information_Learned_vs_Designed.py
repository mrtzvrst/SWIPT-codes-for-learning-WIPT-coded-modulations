import pickle
import numpy as np
import matplotlib.pyplot as plt
import Fun_Lib as Fs
import scipy.io
    
Dict_inf = {}
n = 1 
rate = 4 
Pa = 1 
sequence_l = 1000000
snr_vec_in_dB = np.arange(0,26)
snr_vec = 10**(snr_vec_in_dB/10)
fname = 'SER_vs_SNR_Rate_'+str(rate)+'.pickle'
###############################################################################
k = rate
"Conventional QAM Modulation"
Codebook = Fs.QAM_gen(2**(k/2),2**(k/2),Pa=1)
#plt.plot(Codebook[:,0],Codebook[:,1],'.')
Codebook_norm = np.sum(Codebook**2,axis=1).reshape(1,-1)
Codebook_tran = Codebook.transpose()

err_vec = []
for snr in snr_vec:
    err_rate = Fs.SER_calculator(k,snr,1,sequence_l,Codebook,Codebook_tran,Codebook_norm,101)
    err_vec.append(err_rate)
    print('SNR is: '+str(snr)+', SER: '+str(err_rate))
    
plt.plot(snr_vec_in_dB,10*np.log10(np.asarray(err_vec)),'.-')
#plt.plot(snr_vec_in_dB,10*np.log10(1-erf(np.sqrt(snr_vec/2))),'.-')

Dict_inf['Conventional_16QAM'] = [snr_vec_in_dB,10*np.log10(np.asarray(err_vec))]
with open(fname, 'wb') as f:
    pickle.dump(Dict_inf, f)  
###############################################################################
k = rate
M = 2**k
"Designed Modulation"
X_r, X_i = Fs.Generate_constellation(M, Pa)
    
Codebook_tran, Codebook_norm, Codebook = Fs.Codebook_adjusting(X_r.reshape(-1,1), X_i.reshape(-1,1), n, M, 1)

err_vec = []
for snr in snr_vec:
    err_rate = Fs.SER_calculator(k,snr,1,sequence_l,Codebook,Codebook_tran,Codebook_norm,101)
    err_vec.append(err_rate)
    print('SNR is: '+str(snr)+', SER: '+str(err_rate))
    
plt.plot(snr_vec_in_dB,10*np.log10(np.asarray(err_vec)),'.-')
plt.show()

Dict_inf['Designed_16QAM'] = [snr_vec_in_dB,10*np.log10(np.asarray(err_vec))]
with open(fname, 'wb') as f:
    pickle.dump(Dict_inf, f)  
###############################################################################    

Dict_inf = {}
rate = 6
fname = 'SER_vs_SNR_Rate_'+str(rate)+'.pickle'
###############################################################################
k = rate
"Conventional QAM Modulation"
Codebook = Fs.QAM_gen(2**(k/2),2**(k/2),Pa=1)
#plt.plot(Codebook[:,0],Codebook[:,1],'.')
Codebook_norm = np.sum(Codebook**2,axis=1).reshape(1,-1)
Codebook_tran = Codebook.transpose()

err_vec = []
for snr in snr_vec:
    err_rate = Fs.SER_calculator(k,snr,1,sequence_l,Codebook,Codebook_tran,Codebook_norm,101)
    err_vec.append(err_rate)
    print('SNR is: '+str(snr)+', SER: '+str(err_rate))
    
plt.plot(snr_vec_in_dB,10*np.log10(np.asarray(err_vec)),'.-')
#plt.plot(snr_vec_in_dB,10*np.log10(1-erf(np.sqrt(snr_vec/2))),'.-')

Dict_inf['Conventional_64QAM'] = [snr_vec_in_dB,10*np.log10(np.asarray(err_vec))]
with open(fname, 'wb') as f:
    pickle.dump(Dict_inf, f)  
###############################################################################
k = rate
M = 2**k
"Designed Modulation"
X_r, X_i = Fs.Generate_constellation(M, Pa)
    
Codebook_tran, Codebook_norm, Codebook = Fs.Codebook_adjusting(X_r.reshape(-1,1), X_i.reshape(-1,1), n, M, 1)

err_vec = []
for snr in snr_vec:
    err_rate = Fs.SER_calculator(k,snr,1,sequence_l,Codebook,Codebook_tran,Codebook_norm,101)
    err_vec.append(err_rate)
    print('SNR is: '+str(snr)+', SER: '+str(err_rate))
    
plt.plot(snr_vec_in_dB,10*np.log10(np.asarray(err_vec)),'.-')
plt.show()

Dict_inf['Designed_64QAM'] = [snr_vec_in_dB,10*np.log10(np.asarray(err_vec))]
with open(fname, 'wb') as f:
    pickle.dump(Dict_inf, f)  
###############################################################################    
    
"Save the file in MATLAB"
#Save the parameters in MATLAB
with open('SER_vs_SNR_Rate_6.pickle', 'rb') as f:
    evol = pickle.load(f)
scipy.io.savemat('SER_vs_SNR_Rate_6.mat', mdict={'evol': evol}) 
#Save the parameters in MATLAB
with open('SER_vs_SNR_Rate_4.pickle', 'rb') as f:
    evol = pickle.load(f)
scipy.io.savemat('SER_vs_SNR_Rate_4.mat', mdict={'evol': evol}) 
    
#"Designed Modulation"
#with open(fname, 'rb') as f:
#    CB = pickle.load(f)
#plt.plot(CB['Conventional_QAM'][0][0:15],CB['Conventional_QAM'][1][0:15],'.-')
#plt.plot(CB['Designed_QAM_n2'][0][0:15],CB['Designed_QAM_n2'][1][0:15],'.-')
#plt.plot(CB['Designed_QAM_n4'][0][0:15],CB['Designed_QAM_n4'][1][0:15],'.-')
#plt.plot(CB['Designed_QAM_n6'][0],CB['Designed_QAM_n6'][1],'.-')
#plt.show()


