Python
======
from scipy.stats import norm
import numpy as np
import math as math
import pylab
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


np.seterr(divide='ignore')          #ignore warning division by zero encountered in d1 computation


class Normal_distribution:
    def _init_(self):
        self.a1 = 0.31938153
        self.a2 = -0.356563782
        self.a3 = 1.781477937
        self.a4 = -1.821255978
        self.a5 = 1.330274429

    def Cumulative_normal_distribution(self,x):
        d = 1 / (1 + 0.2316419 * np.abs(x))
        N = 1 - (1 / np.sqrt(2 * 3.1415926) * np.exp(-0.5 * (x * x)) * (self.a1 * d + self.a2 * np.power(d,2) + self.a3 * np.power(d,3) + self.a4 * np.power(d,4) + self.a5 * np.power(d,5)))
        if x>=0:
            return N
        else:
            return 1 - N


class Black_Scholes:
    def __init__(self,S,K,S_b,T,r,v,q):
        self.S = S
        self.K = float(K)
        self.S_b = S_b
        self.T = T
        self.r = float(r)
        self.v = float(v)
        self.q = float(q)

    def d1(self,t):
        if type(t)==float:
            return (np.log(self.S/self.K)+(self.r - self.q +(self.v*self.v)/2.)*(self.T-t))/(self.v*np.sqrt(self.T-t))
        else:
            return (np.log(self.S/self.K)+(self.r - self.q +(self.v*self.v)/2.)*(self.T-t).reshape(len(t),1))/(self.v*np.sqrt(self.T-t).reshape(len(t),1))

    def d2(self,t):
        if type(t)==float:
            return (np.log(self.S/self.K)+(self.r - self.q - (self.v*self.v)/2.)*(self.T-t))/(self.v*np.sqrt(self.T-t))
        else:
            return (np.log(self.S/self.K)+(self.r - self.q - (self.v*self.v)/2.)*(self.T-t).reshape(len(t),1))/(self.v*np.sqrt(self.T-t).reshape(len(t),1))


    def get_optionvalue(self,CallPutFlag, Down_Up, In_Out, t):
        d1 = self.d1(t)
        d2 = self.d2(t)
        if CallPutFlag=='call':
            return self.S*norm.cdf(d1)-self.K*np.exp(-self.r*(self.T-t).reshape(len(t),1))*norm.cdf(d2)
        else:
            return self.K*np.exp(-self.r*(self.T-t).reshape(len(t),1))*norm.cdf(-d2)-self.S*norm.cdf(-d1)


    def get_delta1(self,CallPutFlag,t):
        d1 = self.d1(t)
        d2 = self.d2(t)
        if CallPutFlag=='call':
            return norm.cdf(d1) #np.exp((self.T-t)*self.q)*norm.cdf(d1)
        else:
            return norm.cdf(d1)-1 #np.exp((self.T-t)*self.q)*norm.cdf(d1)-1

    def get_gamma1(self,CallPutFlag,t):
        if type(t)==float:
            d1 = (np.log(self.S/self.K)+(self.r+self.v*self.v/2.)*(self.T-t))/(self.v*np.sqrt(self.T-t))
            a = (self.S*self.v*np.sqrt(self.T-t))
        else:
            d1 = (np.log(self.S/self.K) + (self.r+self.v*self.v/2.)*(self.T-t).reshape(len(t),1))/(self.v*np.sqrt(self.T-t).reshape(len(t),1))
            a = (self.S*self.v*np.sqrt(self.T-t).reshape(len(t),1))
        return norm.pdf(d1)/a #(1/np.sqrt(2*3.1415926))*np.exp(-0.5*d1*d1)/a #/(self.S*self.v*np.sqrt(self.T-t)) #np.exp((self.T-t)*self.q)*norm.pdf(d_1)/(self.S*self.v*np.sqrt(self.T-t))

    def get_delta(self,CallPutFlag,Up_Down,In_Out,t,d=0.0001):
        self.S += d
        after_optionvalue = self.get_optionvalue(CallPutFlag, Down_Up, In_Out, t)
        self.S -= d
        org_optionvalue = self.get_optionvalue(CallPutFlag, Down_Up, In_Out, t)
        return (after_optionvalue - org_optionvalue) / d

    def get_theta(self,CallPutFlag,Up_Down,In_Out,t,d=0.0001):
        self.t += d
        after_optionvalue = self.get_optionvalue(CallPutFlag, Down_Up, In_Out, t)
        self.t -= d
        org_optionvalue = self.get_optionvalue(CallPutFlag, Down_Up, In_Out, t)
        return (after_optionvalue - org_optionvalue) / d*-1

    def get_gamma(self,CallPutFlag,Up_Down,In_Out,t,d=0.0001):
        self.S += d
        after_delta = self.get_delta(CallPutFlag, Down_Up, In_Out, t)
        self.S -= d
        org_delta = self.get_delta(CallPutFlag, Down_Up, In_Out, t)
        return (after_delta - org_delta) / d

