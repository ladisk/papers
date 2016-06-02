# coding: utf-8
"""
A class for identification of damping ratios based on methods discussed in [1].
It offers an implementation of damping identification using the following methods:
    -continuous wavelet transform (CWT)
    -synchrosqueezed continuous wavelet transform (SWT)
    -averaged synchrosqueezed continuous wavelet transform (SWT_avg)
    -proportional synchrosqueezed continuous wavelet transform (SWT_prop)

[1] Marko Mihalec, Janko Slavič and Miha Boltežar,
Synchrosqueezed Wavelet Transform for Damping Identification,
Mechanical Systems and Signal Processing (2016),
10.1016/j.ymssp.2016.05.005.

See also: http://lab.fs.uni-lj.si/ladisk/?what=abstract&ID=171
"""


from scipy import signal
from scipy import special
from scipy import stats
import numpy as np

#2016, cite this paper: http://lab.fs.uni-lj.si/ladisk/?what=abstract&ID=171

class wt_damping_id:
    def __init__(self, x, t, freq, eta, sig=1.):
        """A class used for computation of (synchrosqueezed) continuous wavelet transform and damping identification

        :param x: Signal, 1d array of time series. The same length as t.
        :param t: Time, 1d array at which each signal was recorded. The same length as x.
        :param freq: Frequencies, 1d array of frequencies for which the CWT or SWT is computed.
        :param eta: Frequency modulation parameter, float.
        :param sig: Parameter of Gaussian window width, float. If sig==1 Morlet, else Gabor wavelet is used.
        """
        self.x = x
        self.t = t
        self.sig = sig
        self.eta = eta
        self.freq = freq # [Hz]
        self.res_frek = len(freq)
        self.res_cas = len(t)
        self.casvs = t[-1]-t[0]
        
    @property
    def cwt(self):
        """ Computes contunous wavelet transform of the signal
        :return: CWT, 2d complex array, a CWT over different times and scales.
        """
        gaus1=1./((self.sig**2*np.pi)**(0.25)) # fraction in Eq.(5) in [1]
        pot=2.*self.sig**2
        def valcon(s):
            Psi=s**-0.5*gaus1*np.exp(-(-uar/s)**2./pot-1.j*self.eta*2.*np.pi*(-uar/s))
            return signal.fftconvolve(Psi,self.x,mode='same')
        uar=np.linspace(-self.casvs/2.,self.casvs/2.,self.res_cas) # creating a centered time array
        sar=self.eta/(self.freq) # calculating an array of scales corresponding to frequencies
        Q=np.zeros((self.res_frek,self.res_cas),dtype=complex)
        for i in range (0,self.res_frek):
            Q[i,:]=valcon(sar[i])
        Q /= self.res_cas
        self.Q = Q
        return Q
    
    def prelimfreq(self):
        """ Computes preliminary frequencies without modification
        :return: Preliminary frequency for each point on the time-scale plane, 2d complex array

        See Eq.(6) in [1]
        """
        try:
            self.Q
        except AttributeError:
            self.cwt

        res_cas1=1./(self.res_cas-1.)*self.casvs
        res_frek1=1./(self.res_frek-1.)*(self.freq[-1]-self.freq[0])*2.*np.pi
        gradW=np.gradient(self.Q, res_frek1, res_cas1)[1]
        omega=np.zeros(self.Q.shape,dtype=complex)
        for i in range (0,self.res_frek):
            for ii in range (0,self.res_cas-1):
                if abs(self.Q[i,ii])>1E-10:
                    omega[i,ii]=-1.j*gradW[i,ii]/self.Q[i,ii]
        self.omega = omega
        return omega

    def prelimfreq_scd(self):
        """ Computes preliminary frequencies with scale dependant modification
        :return: Preliminary frequency for each point on the time-scale plane, 2d complex array

        See Section 3.1 in [1], Eq.(28) and Eq.(30).
        """
        try:
            self.Q
        except AttributeError:
            self.cwt

        res_cas1=1./(self.res_cas-1.)*self.casvs
        res_frek1=1./(self.res_frek-1.)*(self.freq[-1]-self.freq[0])*2.*np.pi
        gradW=np.gradient(self.Q,res_frek1,res_cas1)[1]
        omega=np.zeros(self.Q.shape,dtype=complex)
        for i in range (0,self.res_frek):
            for ii in range (0,self.res_cas-1):
                if abs(self.Q[i,ii])>1E-10:
                    omega[i,ii]=-1.j*gradW[i,ii]/self.Q[i,ii]      
    
        war = self.freq*2*np.pi
        frekmatrika = war[:,np.newaxis]*np.ones_like(omega)
        hkorekcijski=frekmatrika*self.casvs/(self.res_cas-1.)
        omega=omega*hkorekcijski/np.sin(hkorekcijski)
        self.omega = omega
        return omega    


    def prelimfreq_shifted(self):
        """ Computes preliminary frequencies with shifted coefficient modification
        :return: Preliminary frequency for each point on the time-scale plane, 2d complex array

        See Section 3.1 in [1], Eq.(28) and Eq.(31).
        """
        try:
            self.Q
        except AttributeError:
            self.cwt

        res_cas1=1./(self.res_cas-1.)*self.casvs
        res_frek1=1./(self.res_frek-1.)*(self.freq[-1]-self.freq[0])*2.*np.pi
        gradW=np.gradient(self.Q,res_frek1,res_cas1)[1]
        omega=np.zeros(self.Q.shape,dtype=complex)
        for i in range (0,self.res_frek):
            for ii in range (0,self.res_cas-1):
                if abs(self.Q[i,ii])>1E-10:
                    omega[i,ii]=-1.j*gradW[i,ii]/self.Q[i,ii]
        
                    korek=omega[i,ii]*self.casvs/(self.res_cas-1)
                    omega[i,ii]=omega[i,ii]*korek/np.sin(korek)

        self.omega = omega
        return omega    
    
    def prelimfreq_auto(self):
        """ Computes preliminary frequencies with autocorrelated modification
        :return: Preliminary frequency for each point on the time-scale plane, 2d complex array

        See Section 3.1 in [1], Eq.(32) and Eq.(33).
        """
        try:
            self.Q
        except AttributeError:
            self.cwt

        res_cas1=1./(self.res_cas-1.)*self.casvs
        res_frek1=1./(self.res_frek-1.)*(self.freq[-1]-self.freq[0])*2.*np.pi
        gradW=np.gradient(self.Q,res_frek1,res_cas1)[1]
        omega=np.zeros(self.Q.shape,dtype=complex)
        for i in range (0,self.res_frek):
            for ii in range (0,self.res_cas-1):
                if abs(self.Q[i,ii])>1E-10:
                    omega[i,ii]=-1.j*gradW[i,ii]/self.Q[i,ii]

                    argument=np.abs(omega[i,ii])*self.casvs/(self.res_cas-1)
                    if argument<=1:
                        sinus=np.arcsin(argument)
                        korek=sinus/np.sin(sinus)
                        omega[i,ii]=omega[i,ii]*korek

        self.omega = omega
        return omega

    
    def swt(self, omegacor=True):
        """ Computes the synchrosqueezed wavelet transform
        :param omegacor: If True, the autocorrelated modification is used to compute preliminary frequencies. If false, preliminary frequencies are used without modification
        :return: SWT, 2d complex array, a SWT over different times and frequencies.
        """
        #Check to see whether Q has already been computed
        try:
            self.Q
        except AttributeError:
            self.cwt
        if omegacor:
            self.prelimfreq_auto()
        else:
            self.prelimfreq()
            
        delomega = (self.freq[11]-self.freq[10])*2*np.pi
        delomega2 = delomega/2
        sar=self.eta/self.freq
        sar23 = (self.eta/self.freq)**(2./3.)
        absomega=np.abs(self.omega)
        
        temp=sar23[1:]*(sar[1:]-sar[:-1])
        T=np.zeros((self.res_frek,self.res_cas),dtype=complex)
        min_freq=np.min(self.freq)*2*np.pi
        max_freq=np.max(self.freq)*2*np.pi
        for ia in range (0,self.res_frek):
            omegal=self.freq[ia]
            for ib in range (0,self.res_cas):
                if min_freq<absomega[ia,ib]<max_freq:
                    __,_ = min(enumerate(self.freq*2*np.pi), key=lambda x: abs(x[1]-absomega[ia,ib]))
                    T[__,ib] += self.Q[ia,ib]*temp[ia-1]/delomega
        self.T = T
        return T
    
    def swt_avg(self, omegacor=True):
        """ Computes the synchrosqueezed wavelet transform using the average SWT criterion
        :param omegacor: If True, the autocorrelated modification is used to compute preliminary frequencies. If false, preliminary frequencies are used without modification
        :return: SWT, 2d complex array, a SWT over different times and frequencies.
        """
        if omegacor:
            self.prelimfreq_auto()
        else:
            self.prelimfreq()
                
        sar=self.eta/self.freq
        delomega=(self.freq[11]-self.freq[10])*2.*np.pi
        delomega2=delomega/2
        sar23=sar**(2./3.)
        absomega=abs(self.omega)
        temp=sar23[1:]*(sar[1:]-sar[:-1])
        T=np.zeros((self.res_frek,self.res_cas),dtype=complex)
        avgomega=np.mean(absomega,axis=1)    
        for ia in range (0,self.res_frek):
            omegal=self.freq[ia]*2*np.pi
            SUMA=np.zeros(self.res_cas)
            for k in range (1,self.res_frek):
                if abs(avgomega[k]-omegal)<=delomega2:
                    SUMA=SUMA+self.Q[k,:]*temp[k-1]
            T[ia,:]=SUMA/delomega
        self.T_avg = T
        return self.T_avg
    
    def swt_prop(self,omegacor=True):
        """ Computes the synchrosqueezed wavelet transform using the proportional SWT criterion
        :param omegacor: If True, the autocorrelated modification is used to compute preliminary frequencies. If false, preliminary frequencies are used without modification
        :return: SWT, 2d complex array, a SWT over different times and frequencies.
        """
        if omegacor:
            self.prelimfreq_auto()
        else:
            self.prelimfreq()
        
        sar=self.eta/self.freq
        delomega=(self.freq[11]-self.freq[10])*2.*np.pi
        delomega2=delomega/2
        sar23=sar**(2./3.)
        absomega=abs(self.omega)
        temp=sar23[1:]*(sar[1:]-sar[:-1])
        T=np.zeros((self.res_frek,self.res_cas),dtype=complex)
        kriterij=np.zeros((self.res_frek,self.res_cas,self.res_frek-1),dtype=np.bool)
        for ia in range (0,self.res_frek):
            omegal=self.freq[ia]*2*np.pi
            for ib in range (0,self.res_cas-1):
                for k in range (1,self.res_frek):
                    if abs(absomega[k,ib]-omegal)<=delomega2:
                        kriterij[ia,ib,k-1]=1
        vsota=np.sum(kriterij,1)
        for ia in range (0,self.res_frek):
            SUMA=np.zeros(self.res_cas)
            for k in range (0,self.res_frek-1):
                SUMA=SUMA+self.Q[k,:]*temp[k]*vsota[ia,k]/(self.res_cas-1.)
            T[ia,:]=SUMA/delomega
        self.T_prop = T
        return self.T_prop
    
    def obrez(self, z1=10, z2=10):
        """ A function that cuts the edges of the transformed signals to avoid the edge effect
        :param z1: percentage of signal that is cutoff at the beginning
        :param z2: percentage of signal that is cut off at the end
        :return: Cut version of all the transforms that have been computed
        """
        meja1=int(self.res_cas*z1/100)
        meja2=int(self.res_cas*(1-z2/100))
        try:
            self.Q
            self.Qcut=self.Q[:,meja1:meja2]
        except AttributeError:
            pass
        try:
            self.T
            self.Tcut=self.T[:,meja1:meja2]
        except AttributeError:
            pass
        try:
            self.T_avg
            self.T_avg_cut=self.T_avg[:,meja1:meja2]
        except AttributeError:
            pass
        try:
            self.T_prop
            self.T_prop_cut=self.T_prop[:,meja1:meja2]
        except AttributeError:
            pass
        self.t_cut=self.t[meja1:meja2]
        
    def skel(self, M1, x01, width=20):
        """ A function that extracts a skeleton from a 2d array, by searching for a maximum of a absolute value within a band around x01.
        :param M1: 2d array from which the
        :param x01: center of the band
        :param width: One half of the width of the band
        :return: Skeleton, 2d array: absolute value on the ridge, zeroes everywhere else.

        x01 is the index of the frequency in 'freq', around which the ridge is searched for.
        x01 must be x01>width and x01<(len(freq)-width) so that the whole search band is within the array
        """
        M=abs(M1)
        (xlen,ylen)=M.shape
        skel=np.zeros((xlen,ylen))
        izhodisce=x01
        for y in range (0,ylen):
            indmax=np.argmax(M[izhodisce-width:izhodisce+width,y])
            skel[indmax+izhodisce-width,y]=M[indmax+izhodisce-width,y]
        return skel
    
    def ident(self, trans, x01, width=20):
        """ Function that identifies damping.
        :param trans: Transformation to be used in the process, string. Either 'cwt', 'swt', 'swt_avg' or 'swt_prop'
        :param x01: center of the band where the ridge is searched for
        :param width: Width of band
        :return: identified frequency, damping ratio

        x01 is the index of the frequency in 'freq', around which the ridge is searched for.
        x01 must be x01>width and x01<(len(freq)-width) so that the whole search band is within the array
        """
        if trans=='cwt':
            self.cwt;
            self.obrez()
            M = self.skel(self.Qcut, x01, width)
        elif trans=='swt':
            self.swt();
            self.obrez()
            M = self.skel(self.Tcut, x01, width)
        elif trans=='swt_avg':
            self.swt_avg();
            self.obrez()
            M = self.skel(self.T_avg_cut, x01, width)
        elif trans=='swt_prop':
            self.swt_prop();
            self.obrez()
            M = self.skel(self.T_prop_cut, x01, width)
        else:
            raise Exception('Unrecognized method')
        # identified modal frequency is the most common frequency on the ridge:
        frekident=self.freq[np.argmax(np.bincount(np.argmax(abs(M),0)))]
        smat = np.zeros(M.shape)
        for i in range (0,smat.shape[1]):
            smat[:,i]=self.eta/self.freq
        pomozna=(2.*abs(M)/((4*np.pi*self.sig**2*smat**2)**(0.25)))
        leva=np.log(np.amax(pomozna,0)) # left hand side of Eq.(9) in [1]
        leva[leva==-np.inf]=0 # Overwriting divisions by 0
        leva[leva==0]=np.mean(leva)
        desna=-1*frekident*2*np.pi*self.t_cut
        DES = np.vstack([desna, np.ones(len(desna))]).T # right hand side of Eq.(9) in [1]
        zeta_iden, lnAmp = np.linalg.lstsq(DES, leva)[0]
        return frekident, zeta_iden
        

if __name__ == "__main__":
    t = np.linspace(0,2,2000) # array of time
    w = 20 # frequency
    z = 0.01 # damping ratio
    x = np.sin(2*np.pi*w*t)*np.exp(-2*np.pi*w*z*t)+(np.random.rand(len(t))-0.5) #signal with added noise: Eq.(43) in [1]

    frequencies = np.linspace(15,25,50) # range of frequencies for which the signal will be analyzed
    WT = wt_damping_id(x,t,frequencies,5)
    freq_id, z_id = WT.ident('swt_prop',25,10)
    print('Identified frequency: {0:4.1f}'.format(freq_id))
    print('Identified damping: {0:4.2}'.format(z_id))