import numpy as np
from numpy.fft import fft
import matplotlib.pyplot as plt
from scipy.signal import TransferFunction as TF

def OSI_var1(u_signal, f_signal, Fs, NT, NT_cancel=5, T=1):
    '''
    Computes the Frequency Response Function (FRF) of an operating system 
    using the Operational System Identification (OSI) method.

    Parameters
    ----------
    u_signal :  ndarray, float
                sampled response signal
    f_signal :  ndarray, float
                sampled force/ excitation signal
    Fs :        scalar, int
                sampling frequency, number of data samples aquired per second
    T  :        scalar, float, optional
                measurement block length in seconds, time interval of the FFT
    NT_cancel : scalar, int
                number of measurement blocks that are dropped in the beginning, 
                time that the sytem needs to reach the steady state 
    NT :        scalar, int
                number of measurement blocks used for the OSI
    
    Returns
    -------
    Y_osi   : ndarray, complex
              FRF determined using the OSI-method
    freq    : ndarray, float
              discrete frequencies associated to the FFT
    U_avg   : ndarray, complex
              FFT of the averaged response signal
    F_avg   : ndarray, complex
              FFT of the averaged force signal
                  
    References:     
    -----------
    [1] de Klerk, D. "Determination of an Operating Systems' Receptance FRF 
        Data (Continued)." In: proceedings on the 26th  International Modal 
        Analysis Conference (IMAC), Orlando, FL
    
    [2] de Klerk, D. "Dynamic Response Characterization of Complex Systems 
        through Operational Identification and Dynamic Substructuring", 
        Dissertation, 2009
      
    See Also:
    --------
    
    Examples:
    --------
    

    '''    
 
    # maybe some helpful relationships:
    
    # time domain:
    T_tot     = NT*T                  # time interval used for the OSI
    T_tot_min = T_tot + NT_cancel*T   # minimum signal length, the first five measurement blocks are dropped
    N         = T*Fs                  # total number of data samples per measurement block
    delta_t   = 1/Fs                  # = T/N ; sampling time
    # frequency domain:
    F_max     = Fs/2.0                # bandwidth, highest frequency captured in the FFT
    delta_f   = 1/T                   # = F_max/SL ; frequency resolution, distance between spectral lines
    SL        = N//2                  # spectral lines, number of samples in freq. domain
    
    # check if signal is long enough:
    if (T_tot_min*N) > len(u_signal): 
        raise ValueError('Signal is too short. Provide a longer signal or decrease \
                          the number of measurement blocks NT or measurement block length T.')
        
    # averaging:
    
    # the OSI-method can be realized in two different ways: FFT of all measure-
    # ment blocks and then averaging in frequency domain or averaging in time 
    # domain and then FFT of the averaged time signal. Here the 2nd variant is 
    # chosen, because averaging of the time signal and performing one FFT is 
    # much more cheaper than performing NT FFTs and averaging in frequency domain.
    
    u_avg = np.zeros(N)    # initialization of the averaged time domain data
    f_avg = np.zeros(N)    
    err_u = np.zeros(NT)  # initialization of convergence vectors
    err_f = np.zeros(NT)
    
    for i in range(0,NT):
        start = (i+NT_cancel)*N        # first and last index of the i-th measurement 
        end   = (i+1+NT_cancel)*N      # block; the first NT_cancel measurement blocks are dropped
        u_block = u_signal[start:end]
        f_block = f_signal[start:end]
        u_avg   += u_block
        f_avg   += f_block
        
        # determination of a convergence measure for the averaged blocks:
        u_abs_i = 1/(i+1) * u_avg             # total value / norm of i averaged measurement blocks
        u_abs_i = np.sum( np.abs(u_abs_i) )
        f_abs_i = 1/(i+1) * f_avg
        f_abs_i = np.sum( np.abs(f_abs_i) )
        
        if i==0:                     # the first measurement block is used as reference for normalization
            u_ref = u_abs_i
            f_ref = f_abs_i
            
        err_u[i] = u_abs_i/u_ref   # relative deviation, normalization
        err_f[i] = f_abs_i/f_ref
                
    u_avg = u_avg/NT
    f_avg = f_avg/NT
    
    # transformation into frequency domain: 
    U_avg = fft(u_avg)/N                            
    U_avg = np.hstack((U_avg[0],2*U_avg[1:SL]))    
    F_avg = fft(f_avg)/N                           
    F_avg = np.hstack((F_avg[0],2*F_avg[1:SL]))
    
    Y_osi = np.divide(U_avg, F_avg)
    freq  = np.arange(0,SL)*delta_f    
    
    # plots:
    
#    # convergence plot of the averaging process
#    plt.semilogx(np.arange(1,NT+1), err_u, 'b', label='response u')
#    plt.semilogx(np.arange(1,NT+1), err_f, 'r', label='force f')
#    plt.grid()
#    plt.ylim(0,1)
#    plt.legend(loc='lower left')
#    plt.xlabel("number of averaged measurement blocks [-]")
#    plt.ylabel("convergence [-]")
#    plt.show()
    
#    # amplitudes of the FRF
#    plt.plot(freq, np.abs(Y_osi))
#    plt.grid()
#    plt.yscale("log")
#    plt.xlabel("frequency [Hz]")
#    plt.ylabel("amplitude [m/s^2]")
#    plt.show()
#    
#    # phase angle of the FRF
#    plt.plot(freq, np.angle(Y_osi,deg=True))
#    plt.grid()
#    plt.xlabel("frequency [Hz]")
#    plt.ylabel("phase [°]")
#    plt.show()
    
    return Y_osi, freq, err_u, err_f




def OSI_var2(u_signal, f_signal, Fs, NT, NT_cancel=5, T=1):
    '''
    Computes the Frequency Response Function (FRF) of an operating system 
    using the Operational System Identification (OSI) method.

    Parameters
    ----------
    u_signal :  ndarray, float
                sampled response signal
    f_signal :  ndarray (vector), float
                sampled force signal
    Fs :        scalar, int
                sampling frequency, number of data samples aquired per second
    T  :        scalar, float, optional
                measurement block length in seconds, time interval of the FFT
    NT_cancel : scalar, int
                number of measurement blocks that are dropped in the beginning, 
                time that the sytem needs to reach the steady state 
    NT :        scalar, int
                number of measurement blocks
    
    Returns
    -------
    Y_osi   : ndarray, complex
              FRF determined using the OSI-method
    freq    : ndarray, float
              discrete frequencies associated to the FFT
    U_avg   : ndarray, complex
              FFT of the averaged response signal
    F_avg   : ndarray, complex
              FFT of the averaged force signal
                  
    References:     
    -----------
    [1] de Klerk, D. "Determination of an Operating Systems' Receptance FRF 
        Data (Continued)." In: proceedings on the 26th  International Modal 
        Analysis Conference (IMAC), Orlando, FL
    
    [2] de Klerk, D. "Dynamic Response Characterization of Complex Systems 
        through Operational Identification and Dynamic Substructuring", 
        Dissertation, 2009
      
    See Also:
    --------
    
    Examples:
    --------
    

    '''    
 
    # maybe some helpful relationships:
    
    # time domain:
    T_tot     = NT*T                  # time interval used for the OSI
    T_tot_min = T_tot + NT_cancel*T   # minimum signal length, the first five measurement blocks are dropped
    N         = T*Fs                  # total number of data samples per measurement block
    delta_t   = 1/Fs                  # = T/N ; sampling time
    # frequency domain:
    F_max     = Fs/2.0        # bandwidth, highest frequency captured in the FFT
    delta_f   = 1/T           # = F_max/SL ; frequency resolution, distance between spectral lines
    SL        = N//2          # number of spectral lines, number of samples in freq. domain
    
    # check if signal is long enough:
    if (T_tot_min*N) > len(u_signal): 
        raise ValueError('Signal is too short. Provide a longer signal or decrease \
                          the number of measurement blocks NT or measurement block length T.')
        
    # averaging:
    
    # the OSI-method can be realized in two different ways: FFT of all measurement blocks and then averaging in frequency 
    # domain or averaging in time domain and then FFT of the averaged time signal. Here the 1st variant is chosen.
    
    U_avg = np.zeros(SL,dtype=complex) # initialization of the averaged frequency domain data
    F_avg = np.zeros(SL,dtype=complex)  
    err_U = np.zeros(NT)               # initialization of convergence vectors
    err_F = np.zeros(NT)
    
    for i in range(0,NT):
        start = (i+NT_cancel)*N        # first and last index of the i-th measurement 
        end   = (i+1+NT_cancel)*N      # block; the first five measurement blocks are dropped
        u_block = u_signal[start:end]
        f_block = f_signal[start:end]

        # transformation into frequency domain: 
        U_block = fft(u_block)/N                  
        U_block = np.hstack((U_block[0],2*U_block[1:SL]))         
        F_block = fft(f_block)/N                  
        F_block = np.hstack((F_block[0],2*F_block[1:SL]))

        U_avg   += U_block
        F_avg   += F_block
        
        # determination of a convergence measure for the averaged blocks:
        U_abs_i = 1/(i+1) * U_avg             # total value / norm of i averaged measurement blocks
        U_abs_i = np.sum( np.abs(U_abs_i) )
        F_abs_i = 1/(i+1) * F_avg
        F_abs_i = np.sum( np.abs(F_abs_i) )
        
        if i==0:                     # the first measurement block is used as reference for normalization
            U_ref = U_abs_i
            F_ref = F_abs_i
            
        err_U[i] = U_abs_i/U_ref   # relative deviation, normalization
        err_F[i] = F_abs_i/F_ref
        
    U_avg = U_avg/NT
    F_avg = F_avg/NT
    
    Y_osi = np.divide(U_avg, F_avg)
    freq  = np.arange(0,SL)*delta_f  
    
    # plots:
    
#    # convergence plot of the averaging process
#    plt.plot(np.arange(1,NT+1), err_U, 'b', label='response u')
#    plt.plot(np.arange(1,NT+1), err_F, 'r', label='force f')
#    plt.grid()
#    plt.legend(loc='lower left')
#    plt.xlabel("number of averaged measurement blocks [-]")
#    plt.ylabel("convergence [-]")
#    plt.show()
    
#    # amplitudes of the FRF
#    plt.plot(freq, np.abs(Y_osi))
#    plt.grid()
#    plt.xlabel("frequency [Hz]")
#    plt.ylabel("amplitude [m/s^2]")
#    plt.show()
#    
#    # phase angle of the FRF
#    plt.plot(freq, np.angle(Y_osi,deg=True))
#    plt.grid()
#    plt.xlabel("frequency [Hz]")
#    plt.ylabel("phase [°]")
#    plt.show()
    
    return Y_osi, freq, err_U, err_F

def MIMO_OSI(responses,excitations, Fs_=4096, NT_=300, NT_cancel_=5, T_=1):
    '''
    Application of the OSI method for multiple inputs and outputs

    Parameters
    ----------
    responses :   ndarray, float
                  sampled response signal of the accelerometers
                  1st dim: samples; 2nd dim: sensor channels, 3rd dim: excitation at different locations / different excitation points
                                                                      has to match with the 2nd dim of excitations
    excitations : ndarray, float
                  sampled force signal of the force transducer / impedance sensor
                  1st dim: samples; 2nd dim: excitation at different locations / different excitation points
    Fs_ :         scalar, int
                  sampling frequency, number of data samples aquired per second
    T_  :         scalar, float, optional
                  measurement block length in seconds, time interval to perform a FFT
    NT_cancel_ :  scalar, int
                  number of measurement blocks that are dropped in the beginning, 
                  time that the sytem needs to reach the steady state 
    NT_ :         scalar, int
                  number of measurement blocks

    Returns
    -------
    Y_uf   : ndarray, complex
             FRF matrix determined using the OSI-method
             1st dim: frequency response, 2nd dim: output channel / index, 3rd dim: input channel/ index 
    freq   : ndarray, float
             discrete frequencies associated to the FRF
                  
    References:     
    -----------

    See Also:
    --------
    
    Examples:
    --------
    
    '''    
    # check the correct dimensions:
    dim1_resp, dim2_resp, dim3_resp = np.shape(responses) 
    dim1_exc,  dim2_exc             = np.shape(excitations)
    
    if dim1_resp != dim1_exc: 
        raise ValueError('Amount of time samples of response and excitation signals do not match!')
        
    if dim3_resp != dim2_exc: 
        raise ValueError('Response array and excitation array have a different number of excitation locations!')

    for i in range(0,dim3_resp):
        for j in range(0,dim2_resp):
            
            response = responses[:,j,i]
            force    = excitations[:,i]
            Y_osi, freq, _, _ = OSI_var1(response, force, Fs=Fs_, NT=NT_, NT_cancel=NT_cancel_, T=T_)
            
            if i==0 and j==0:
                Y_uf = np.zeros((np.size(freq),dim2_resp,dim3_resp),dtype=complex)
                
            Y_uf[:,j,i] = Y_osi

    return freq, Y_uf
