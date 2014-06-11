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



class Barrier_option:
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


    def d3(self,t):
        if type(t)==float:
            return (np.log(self.S/self.S_b)+(self.r - self.q + (self.v*self.v)/2.)*(self.T-t))/(self.v*np.sqrt(self.T-t))
        else:
            return (np.log(self.S/self.S_b)+(self.r - self.q + (self.v*self.v)/2.)*((self.T-t).reshape(len(t),1)))/(self.v*np.sqrt(self.T-t).reshape(len(t),1))


    def d4(self,t):
        if type(t)==float:
            return (np.log(self.S/self.S_b)+(self.r - self.q - (self.v*self.v)/2.)*(self.T-t))/(self.v*np.sqrt(self.T-t))
        else:
            return (np.log(self.S/self.S_b)+(self.r - self.q - (self.v*self.v)/2.)*(self.T-t).reshape(len(t),1))/(self.v*np.sqrt(self.T-t).reshape(len(t),1))

    def d5(self,t):
        if type(t)==float:
            return (np.log(self.S/self.S_b)-(self.r - self.q - (self.v*self.v)/2.)*(self.T-t))/(self.v*np.sqrt(self.T-t))
        else:
            return (np.log(self.S/self.S_b)-(self.r - self.q - (self.v*self.v)/2.)*(self.T-t).reshape(len(t),1))/(self.v*np.sqrt(self.T-t).reshape(len(t),1))

    def d6(self,t):
        if type(t)==float:
            return (np.log(self.S/self.S_b)-(self.r - self.q + (self.v*self.v)/2.)*(self.T-t))/(self.v*np.sqrt(self.T-t))
        else:
            return (np.log(self.S/self.S_b)-(self.r - self.q + (self.v*self.v)/2.)*(self.T-t).reshape(len(t),1))/(self.v*np.sqrt(self.T-t).reshape(len(t),1))

    def d7(self,t):
        if type(t)==float:
            return (np.log((self.S*self.K)/(self.S_b*self.S_b))-(self.r - self.q - (self.v*self.v)/2.)*(self.T-t))/(self.v*np.sqrt(self.T-t))
        else:
            return (np.log((self.S*self.K)/(self.S_b*self.S_b))-(self.r - self.q - (self.v*self.v)/2.)*(self.T-t).reshape(len(t),1))/(self.v*np.sqrt(self.T-t).reshape(len(t),1))

    def d8(self,t):
        if type(t)==float:
            return (np.log((self.S*self.K)/(self.S_b*self.S_b))-(self.r - self.q + (self.v*self.v)/2.)*(self.T-t))/(self.v*np.sqrt(self.T-t))
        else:
            return (np.log((self.S*self.K)/(self.S_b*self.S_b))-(self.r - self.q + (self.v*self.v)/2.)*(self.T-t).reshape(len(t),1))/(self.v*np.sqrt(self.T-t).reshape(len(t),1))

    def a(self,t):
        if type(t)==float:
            return np.power((self.S_b / self.S), (-1 + 2 * (self.r - self.q) / (self.v*self.v)))
        else:
            return np.power((self.S_b / self.S), (-1 + 2 * (self.r - self.q) / (self.v*self.v))).reshape(len(t),len(self.S[0]))


    def b(self,t):
        if type(t)==float:
            return np.power((self.S_b / self.S), (1 + 2 * (self.r - self.q) / (self.v*self.v)))
        else:
            return np.power((self.S_b / self.S), (1 + 2 * (self.r - self.q) / (self.v*self.v))).reshape(len(t),len(self.S[0]))

    def get_optionvalue(self,CallPutFlag, Down_Up, In_Out, t):
       if Down_Up == 'Down':
           if In_Out == 'In':
               return self.Down_And_In(CallPutFlag,t)
           else:
               return self.Down_And_Out(CallPutFlag,t)
       else:
             if In_Out == 'In':
               return self.Up_And_In(CallPutFlag,t)
             else:
               return self.Up_And_Out(CallPutFlag,t)


    def Up_And_Out(self,CallPutFlag,t):
        d_1 = self.d1(t)
        d_2 = self.d2(t)
        d_3 = self.d3(t)
        d_4 = self.d4(t)
        d_5 = self.d5(t)
        d_6 = self.d6(t)
        d_7 = self.d7(t)
        d_8 = self.d8(t)
        a = self.a(t)
        b = self.b(t)
        if type(t)==float:
            if CallPutFlag=='call':
               return self.S * np.exp(-self.q * (self.T-t)) * (norm.cdf(d_1) - norm.cdf(d_3) - b * (norm.cdf(d_6) - norm.cdf(d_8))) - self.K * np.exp(-self.r * (self.T-t)) * (norm.cdf(d_2) - norm.cdf(d_4) - a * (norm.cdf(d_5) - norm.cdf(d_7)))
            else:
                if self.K>self.S_b:
                   return -self.S * np.exp(-self.q * (self.T-t)) * (1 - norm.cdf(d_3) - b * norm.cdf(d_6)) + self.K * np.exp(-self.r * (self.T-t)) * (1 - norm.cdf(d_4) - a * norm.cdf(d_5))
                else:
                   return -self.S * np.exp(-self.q * (self.T-t)) * (1 - norm.cdf(d_1) - b * norm.cdf(d_8)) + self.K * np.exp(-self.r * (self.T-t)) * (1 - norm.cdf(d_2) - a * norm.cdf(d_7))
        else:
            if CallPutFlag=='call':
               return self.S * np.exp(-self.q * (self.T-t).reshape(len(t),1)) * (norm.cdf(d_1) - norm.cdf(d_3) - b * (norm.cdf(d_6) - norm.cdf(d_8))) - self.K * np.exp(-self.r * (self.T-t).reshape(len(t),1)) * (norm.cdf(d_2) - norm.cdf(d_4) - a * (norm.cdf(d_5) - norm.cdf(d_7)))
            else:
                if self.K>self.S_b:
                   return -self.S * np.exp(-self.q * (self.T-t).reshape(len(t),1)) * (1 - norm.cdf(d_3) - b * norm.cdf(d_6)) + self.K * np.exp(-self.r * (self.T-t).reshape(len(t),1)) * (1 - norm.cdf(d_4) - a * norm.cdf(d_5))
                else:
                   return -self.S * np.exp(-self.q * (self.T-t).reshape(len(t),1)) * (1 - norm.cdf(d_1) - b * norm.cdf(d_8)) + self.K * np.exp(-self.r * (self.T-t).reshape(len(t),1)) * (1 - norm.cdf(d_2) - a * norm.cdf(d_7))


    def Up_And_In(self,CallPutFlag,t):
        d_1 = self.d1(t)
        d_2 = self.d2(t)
        d_3 = self.d3(t)
        d_4 = self.d4(t)
        d_5 = self.d5(t)
        d_6 = self.d6(t)
        d_7 = self.d7(t)
        d_8 = self.d8(t)
        a = self.a(t)
        b = self.b(t)
        if type(t)==float:
            if CallPutFlag=='call':
               return self.S * np.exp(-self.q * (self.T-t)) * (norm.cdf(d_3) + b * (norm.cdf(d_6) - norm.cdf(d_8))) - self.K * np.exp(-self.r * (self.T-t)) * (norm.cdf(d_4) + a * (norm.cdf(d_5) - norm.cdf(d_7)))
            else:
                if self.K > self.S_b:
                    return -self.S * np.exp(-self.q * (self.T-t)) * (norm.cdf(d_3) - norm.cdf(d_1) + b * norm.cdf(d_6)) + self.K * np.exp(-self.r * (self.T-t)) * (norm.cdf(d_4) - norm.cdf(d_2) + a * norm.cdf(d_5))
                else:
                    return -self.S * np.exp(-self.q * (self.T-t)) * b * norm.cdf(d_8) + self.K * np.exp(-self.r * (self.T-t)) * a * norm.cdf(d_7)
        else:
            if CallPutFlag=='call':
               return self.S * np.exp(-self.q * (self.T-t).reshape(len(t),1)) * (norm.cdf(d_3) + b * (norm.cdf(d_6) - norm.cdf(d_8))) - self.K * np.exp(-self.r * (self.T-t).reshape(len(t),1)) * (norm.cdf(d_4) + a * (norm.cdf(d_5) - norm.cdf(d_7)))
            else:
                if self.K > self.S_b:
                    return -self.S * np.exp(-self.q * (self.T-t).reshape(len(t),1)) * (norm.cdf(d_3) - norm.cdf(d_1) + b * norm.cdf(d_6)) + self.K * np.exp(-self.r * (self.T-t).reshape(len(t),1)) * (norm.cdf(d_4) - norm.cdf(d_2) + a * norm.cdf(d_5))
                else:
                    return -self.S * np.exp(-self.q * (self.T-t).reshape(len(t),1)) * b * norm.cdf(d_8) + self.K * np.exp(-self.r * (self.T-t).reshape(len(t),1)) * a * norm.cdf(d_7)


    def Down_And_Out(self,CallPutFlag,t):
        d_1 = self.d1(t)
        d_2 = self.d2(t)
        d_3 = self.d3(t)
        d_4 = self.d4(t)
        d_5 = self.d5(t)
        d_6 = self.d6(t)
        d_7 = self.d7(t)
        d_8 = self.d8(t)
        a = self.a(t)
        b = self.b(t)
        if type(t)==float:
            if CallPutFlag=='call':
                if self.K > self.S_b:
                    price = self.S * np.exp(-self.q * (self.T-t)) * (norm.cdf(d_1) - b * (1 - norm.cdf(d_8))) - self.K * np.exp(-self.r * (self.T-t)) * (norm.cdf(d_2) - a * (1 - norm.cdf(d_7)))

                else:
                    price = self.S * np.exp(-self.q * (self.T-t)) * (norm.cdf(d_3) - b * (1 - norm.cdf(d_6))) - self.K * np.exp(-self.r * (self.T-t)) * (norm.cdf(d_4) - a * (1 - norm.cdf(d_5)))
            else:
                price = -self.S * np.exp(-self.q * (self.T-t)) * (norm.cdf(d_3) - norm.cdf(d_1) - b * (norm.cdf(d_8) - norm.cdf(d_6))) + self.K * np.exp(-self.r * (self.T-t)) * (norm.cdf(d_4) - norm.cdf(d_2) - a * (norm.cdf(d_7) - norm.cdf(d_5)))
            return max(price,0)
        else:
            if CallPutFlag=='call':
                if self.K > self.S_b:
                    price = self.S * np.exp(-self.q * (self.T-t).reshape(len(t),1)) * (norm.cdf(d_1) - b * (1 - norm.cdf(d_8))) - self.K * np.exp(-self.r * (self.T-t).reshape(len(t),1)) * (norm.cdf(d_2) - a * (1 - norm.cdf(d_7)))
                else:
                    price = self.S * np.exp(-self.q * (self.T-t).reshape(len(t),1)) * (norm.cdf(d_3) - b * (1 - norm.cdf(d_6))) - self.K * np.exp(-self.r * (self.T-t).reshape(len(t),1)) * (norm.cdf(d_4) - a * (1 - norm.cdf(d_5)))
            else:
                price = -self.S * np.exp(-self.q * (self.T-t).reshape(len(t),1)) * (norm.cdf(d_3) - norm.cdf(d_1) - b * (norm.cdf(d_8) - norm.cdf(d_6))) + self.K * np.exp(-self.r * (self.T-t).reshape(len(t),1)) * (norm.cdf(d_4) - norm.cdf(d_2) - a * (norm.cdf(d_7) - norm.cdf(d_5)))
            #print price

        final_price_flow = np.zeros([len(self.S),len(self.S[1])],dtype=float)
        for i in range(1,len(self.S)):
            for j in range(len(self.S[1])):
                if self.S[0,0]<self.S_b:
                    final_price_flow[0,j]=0
                else:
                    final_price_flow[0,j]=price[0,j]
                if (self.S[i,j]<self.S_b) or (final_price_flow[i-1,j]<=0):
                    final_price_flow[i,j]=0.
                else:
                    final_price_flow[i,j] = price[i,j]
        return final_price_flow


    def Down_And_In(self,CallPutFlag,t):
        d_1 = self.d1(t)
        d_2 = self.d2(t)
        d_3 = self.d3(t)
        d_4 = self.d4(t)
        d_5 = self.d5(t)
        d_6 = self.d6(t)
        d_7 = self.d7(t)
        d_8 = self.d8(t)
        a = self.a(t)
        b = self.b(t)
        if type(t)==float:
            if CallPutFlag=='call':
                if self.K > self.S_b:
                    return self.S * np.exp(-self.q * (self.T-t)) * b * (1 - norm.cdf(d_8)) - self.K * np.exp(-self.r * (self.T-t)) * a * (1 - norm.cdf(d_7))
                else:
                    return self.S * np.exp(-self.q * (self.T-t)) * (norm.cdf(d_1) - norm.cdf(d_3) + b * (1 - norm.cdf(d_6))) - self.K * np.exp(-self.r * (self.T-t)) * (norm.cdf(d_2) - norm.cdf(d_4) + a * (1 - norm.cdf(d_5)))
            else:
                return -self.S * np.exp(-self.q * (self.T-t)) * (1 - norm.cdf(d_3) + b * (norm.cdf(d_8) - norm.cdf(d_6))) + self.K * np.exp(-self.r * (self.T-t)) * (1 - norm.cdf(d_4) + a * (norm.cdf(d_7) - norm.cdf(d_5)))
        else:
            if CallPutFlag=='call':
                if self.K > self.S_b:
                    return self.S * np.exp(-self.q * (self.T-t).reshape(len(t),1)) * b * (1 - norm.cdf(d_8)) - self.K * np.exp(-self.r * (self.T-t).reshape(len(t),1)) * a * (1 - norm.cdf(d_7))
                else:
                    return self.S * np.exp(-self.q * (self.T-t).reshape(len(t),1)) * (norm.cdf(d_1) - norm.cdf(d_3) + b * (1 - norm.cdf(d_6))) - self.K * np.exp(-self.r * (self.T-t).reshape(len(t),1)) * (norm.cdf(d_2) - norm.cdf(d_4) + a * (1 - norm.cdf(d_5)))
            else:
                return -self.S * np.exp(-self.q * (self.T-t).reshape(len(t),1)) * (1 - norm.cdf(d_3) + b * (norm.cdf(d_8) - norm.cdf(d_6))) + self.K * np.exp(-self.r * (self.T-t).reshape(len(t),1)) * (1 - norm.cdf(d_4) + a * (norm.cdf(d_7) - norm.cdf(d_5)))


    def get_delta1(self,CallPutFlag,Up_Down,In_Out,t,d=0.0001):
        self.S += d
        after_optionvalue = self.get_optionvalue(CallPutFlag, Down_Up, In_Out, t)
        self.S -= d
        org_optionvalue = self.get_optionvalue(CallPutFlag, Down_Up, In_Out, t)
        return (after_optionvalue - org_optionvalue) / d

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





class StockPrice:
    def __init__(self,r,v,q,dt):
        self.S = float(S)
        self.r = float(r)
        self.v = float(v)
        self.q = float(q)
        self.dt = float(dt)

    def GBM(self,S):
        BM = norm.ppf(np.random.rand())*math.sqrt(self.dt)

        drift = (self.r-self.q-self.v*self.v/2.)*self.dt
        volatility = self.v*BM

        return S*math.exp(drift+volatility)


CallPut = 'call'
Down_Up = 'Down'
In_Out = 'Out'

plot_assetpaths = 'no'
adapt_delta = 'True'
hedge = 'delta'
S = 40.
mu = 0.
RFR = 0.01
vol_actual = 0.2
q = 0
S_b = 35 #30

K_1 = 30
r_1 = 0.01
v_1 = 0.2
q_1 = 0

S_b_limit = 1.
S_b_alternative = S_b_limit + S_b

K_2 = 39
r_2 = 0.01
v_2 = 0.2
q_2 = 0


time_step = 21. #52. #26. #52. #365.*(2/3.) #13. #100 #21.
T = 3/52. #26/52. #21/365. #1. #3/52.
T_alternative = 2*T #T+5*dt
T_limit = T-5/365.
nr_sims = 50000
dt = T/time_step     #time in weeks
nr_shares = 1
fixed_fee = 0.0
perc_up_move = 0.


nr_simdays = int(T/dt)
time_steps = dt*np.array(range(0,nr_simdays+1))
time_to_expiry = T-time_steps

#time_step_orig = 52.
#dt_orig = T/time_step_orig
#nr_simdays_orig = int(T/dt_orig)
#time_steps_orig = dt_orig*np.array(range(0,nr_simdays_orig+1))


#simulation stock
stock = np.zeros([nr_simdays+1,nr_sims],dtype=float)
SP=StockPrice(mu,vol_actual,q,dt)
for j in range(nr_sims):
    stock[0,j] = S
    for i in range(1,nr_simdays+1):
        stock[i,j]=SP.GBM(stock[i-1,j])
    if plot_assetpaths == 'yes':
        pylab.figure(1)
        pylab.plot(time_steps,stock[:,j])
pylab.show()


Option_1 = Barrier_option(stock,K_1,S_b,T,r_1,v_1,q_1) #Black_Scholes(stock,K_1,S_b,T,r_1,v_1,q_1)
alternative_option_1 = Barrier_option(stock,K_1,S_b_alternative,T,r_1,v_1,q_1)
alternative_option_2 = Barrier_option(stock,K_1,S_b,T_alternative,r_1,v_1,q_1)
Option_2 = Black_Scholes(stock,K_2,S_b,T,r_2,v_2,q_2)


if hedge == 'gamma':
    option_flow_1 = np.zeros([nr_simdays+1,nr_sims],dtype=float)
    option_flow_2 = np.zeros([nr_simdays+1,nr_sims],dtype=float)

    # Obtain option flow
    option_value_1 = Option_1.get_optionvalue(CallPut, Down_Up, In_Out, time_steps)
    option_value_2 = Option_2.get_optionvalue(CallPut, Down_Up, In_Out, time_steps)
    option_flow_1[0] = option_value_1[0,]
    option_flow_2[0] = option_value_2[0,]
    for i in range(nr_sims):
        if stock[nr_simdays,i]>K_1:
            option_flow_1[nr_simdays,i] = -max(stock[nr_simdays,i]-K_1,0)
        else:
            option_flow_1[nr_simdays,i]=0
        if stock[nr_simdays,i]>K_2:
            option_flow_2[nr_simdays,i] = -max(stock[nr_simdays,i]-K_2,0)
        else:
            option_flow_2[nr_simdays,i]=0

# -np.fmax(stock[nr_simdays,:]-K_1,0)

    # Prepare delta and gamma options
    delta_1 = Option_1.get_delta(CallPut, Down_Up, In_Out,time_steps)
    delta_2 = Option_2.get_delta(CallPut, Down_Up, In_Out,time_steps)
    gamma_1 = Option_1.get_gamma(CallPut, Down_Up, In_Out,time_steps)
    gamma_2 = Option_2.get_gamma(CallPut, Down_Up, In_Out,time_steps)

    #gamma_1[nr_simdays]=Barrier_option(stock[nr_simdays],K_1,S_b,0.0000003,r_1,v_1,q_1).get_gamma(CallPut, Down_Up, In_Out,0.0)
    #gamma_2[nr_simdays]=Option_2(stock[nr_simdays],K_2,S_b,0.0000003,r_2,v_2,q_2).get_gamma(CallPut, Down_Up, In_Out,0.0)
    Option_1.get_gamma(CallPut, Down_Up, In_Out,0.0)

    # Gamma hedging
    gamma_hedge_ratio = np.zeros([nr_simdays+1,nr_sims],dtype=float)

    gamma_hedge_ratio = -(gamma_1/gamma_2)
    gamma_hedge_ratio[np.where(gamma_2==0)[0]]=0

    hedge_value_option_2 = option_value_2 * gamma_hedge_ratio
    flow_option_2 = np.zeros([nr_simdays+1,nr_sims],dtype=float)
    flow_option_2[0] = hedge_value_option_2[0]
    flow_option_2[nr_simdays] = -hedge_value_option_2[nr_simdays]

    for i in range(nr_simdays-1):
        flow_option_2[i+1] = (gamma_hedge_ratio[i+1] - gamma_hedge_ratio[i])*option_value_2[i+1]


    # Delta hedging
    delta_hedge_ratio = (delta_1 + delta_2*gamma_hedge_ratio)
    hedge_value_asset = -1*stock*delta_hedge_ratio
    flow_asset = np.zeros([nr_simdays+1,nr_sims],dtype=float)
    flow_asset[0] = hedge_value_asset[0]
    flow_asset[nr_simdays] = -hedge_value_asset[nr_simdays]

    for i in range(nr_simdays-1):
        flow_asset[i+1] = -1*(delta_hedge_ratio[i+1] - delta_hedge_ratio[i])*stock[i+1]


    cash_account = np.zeros([nr_simdays+1,nr_sims],dtype=float)
    funding = np.zeros([nr_simdays+1,nr_sims],dtype=float)
    cash_account_without_funding = option_flow_1 + flow_option_2 + flow_asset
    funding[0] = cash_account_without_funding[0] * (math.exp(RFR*dt)-1)
    cash_account[0] = cash_account_without_funding[0]
    cash_account[1] = funding[0] + cash_account_without_funding[1] + cash_account[0]
    funding[1] = (cash_account[0] + cash_account_without_funding[1]) * (math.exp(RFR*dt)-1)

    for i in range(2,nr_simdays+1):
        funding[i] = (cash_account[i-1] + cash_account_without_funding[i]) * (math.exp(RFR*dt)-1)
        cash_account[i] = cash_account[i-1] + funding[i-1] + cash_account_without_funding[i]

    P_L = cash_account_without_funding + funding
    Total_P_L = sum(P_L)

else:
    option_flow_1 = np.zeros([nr_simdays+1,nr_sims],dtype=float)

    # Obtain option flow

    option_value_1 = Option_1.get_optionvalue(CallPut, Down_Up, In_Out, time_steps)

    alternative_option_value_1 = alternative_option_1.get_optionvalue(CallPut, Down_Up, In_Out, time_steps)
    #test = np.where(option_value_1[:,:]<0)
    #option_value_1[np.min(test)]


    option_flow_1[0] = option_value_1[0,]
    #option_flow_1[nr_simdays,:]=-np.fmax(stock[nr_simdays,:]-K_1,0)
    for i in range(nr_sims):
        if stock[nr_simdays,i]>K_1 and option_value_1[nr_simdays-1,i]>0 and stock[nr_simdays,i]>S_b:
            option_flow_1[nr_simdays,i] = -max(stock[nr_simdays,i]-K_1,0)
        else:
            option_flow_1[nr_simdays,i]=0


    # Prepare delta and gamma options
    delta_1 = Option_1.get_delta(CallPut, Down_Up, In_Out,time_steps)
    #delta_1 = alternative_option_1.get_delta(CallPut, Down_Up, In_Out,time_steps)
    delta_orig = Option_1.get_delta(CallPut, Down_Up, In_Out,time_steps)
    delta_alternative = alternative_option_1.get_delta(CallPut, Down_Up, In_Out,time_steps)
    delta_alternative_time = alternative_option_2.get_delta(CallPut, Down_Up, In_Out,time_steps)
    gamma_1 = Option_1.get_gamma(CallPut, Down_Up, In_Out,time_steps)



    # Delta hedging
    delta_hedge_ratio = np.zeros([nr_simdays+1,nr_sims],dtype=float)
    if adapt_delta == 'True':
        for i in range(nr_simdays+1):
            for j in range(nr_sims):

                if stock[i,j] - S_b < S_b_limit:
                    if time_steps[i-1]>T_limit:
                        delta_hedge_ratio[i,j] = delta_alternative[i,j] #0.5*delta_1[i,j] #delta_hedge_ratio[i-1,j] #0 #delta_alternative[i,j]# delta_hedge_ratio[i-1,j] #delta_hedge_ratio[i-1,j] # delta_alternative[i,j] #0.5*delta_1[i,j]
                    else:
                        delta_hedge_ratio[i,j] = delta_alternative[i,j] #0.5*delta_1[i,j] #delta_hedge_ratio[i-1,j] #0 #delta_alternative[i,j] #delta_hedge_ratio[i-1,j] #delta_hedge_ratio[i-1,j] #delta_alternative[i,j] #delta_hedge_ratio[i-1,j]
                else:
                        delta_hedge_ratio[i,j] = delta_alternative[i,j] #delta_1[i,j]
    else:
        delta_hedge_ratio = delta_1





    hedge_value_asset = -stock*delta_hedge_ratio
    flow_asset = np.zeros([nr_simdays+1,nr_sims],dtype=float)
    flow_asset[0] = hedge_value_asset[0]
    flow_asset[nr_simdays] = stock[nr_simdays,:]*delta_1[nr_simdays,:]

    for i in range(nr_simdays-1):
        flow_asset[i+1] = -(delta_hedge_ratio[i+1] - delta_hedge_ratio[i])*stock[i+1]


    cash_account = np.zeros([nr_simdays+1,nr_sims],dtype=float)
    funding = np.zeros([nr_simdays+1,nr_sims],dtype=float)
    fee = np.zeros([nr_simdays+1,nr_sims],dtype=float)
    cash_account_without_funding = option_flow_1 + flow_asset
    funding[0] = cash_account_without_funding[0] * (np.exp(RFR*dt)-1)
    cash_account[0] = cash_account_without_funding[0]
    cash_account[1] = funding[0] + cash_account_without_funding[1] + cash_account[0]
    funding[1] = (cash_account[0] + cash_account_without_funding[1]) * (np.exp(RFR*dt)-1)
    fee[0] = -abs(flow_asset[0])*fixed_fee
    fee[nr_simdays] = -abs(flow_asset[nr_simdays])*fixed_fee*flow_asset[nr_simdays]

    for i in range(2,nr_simdays+1):
        funding[i] = (cash_account[i-1] + cash_account_without_funding[i]) * (np.exp(RFR*dt)-1)
        cash_account[i] = cash_account[i-1] + funding[i-1] + cash_account_without_funding[i]
        for j in range(nr_sims):
            if abs(flow_asset[i-2,j])>0:
                fee[i-1,j] = -abs(flow_asset[i-1,j])*fixed_fee*(1+(perc_up_move*abs(flow_asset[i-1,j]/flow_asset[i-2,j] - 1)))
            else:
                fee[i-1,j] = -abs(flow_asset[i-1,j])*fixed_fee

    P_L = cash_account_without_funding + funding + fee
    Total_P_L = sum(P_L)



start_bin = int(math.floor(min(Total_P_L))) end_bin =  int(math.ceil(max(Total_P_L)))

# Distribution hedging error
#pylab.figure(2)
#n, bins, pathes = pylab.hist(Total_P_L,bins=range(start_bin,end_bin,1), histtype='stepfilled',stacked=True,facecolor='b',cumulative=False)
#n, bins, pathes = pylab.hist(Total_P_L,bins=range(start_bin,-10,1), histtype='bar')

fig, ax = plt.subplots(1)
#ax.hist(Total_P_L,bins=range(start_bin,end_bin,1), histtype='bar',facecolor=[1.,0.501,0.125], cumulative=False,edgecolor=[1.,0.501,0.125])
#ax.hist(Total_P_L,bins=100, histtype='bar',facecolor=[1.,0.501,0.125], cumulative=False,edgecolor=[1.,0.501,0.125])
ax.hist(Total_P_L,bins=range(start_bin,-1,1), histtype='bar',facecolor=[1.,0.501,0.125], cumulative=False,edgecolor=[1.,0.501,0.125], normed=False) #plt.axvline(Total_P_L.mean(), color=[0.321,0.431,0.458], linestyle='dashed', linewidth=1) ax.set_xlabel('hedging error')
ax.set_ylabel('frequency')
pylab.show()

