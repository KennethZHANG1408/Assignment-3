# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 20:35:38 2019

This is a program created by Kenneth for Assignment3
@author: asus
"""

#Prepare for the solutions
import os
import pandas as pd
import numpy as np
import random
import scipy.stats as st
from scipy.optimize import minimize
from scipy.optimize import basinhopping
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
os.chdir('E:\\Desktop\\2019_Spring\\Microstructure\\Assignment3')
data_dir = 'E:\\Desktop\\2019_Spring\\Microstructure\\SH600519'
dataset = pd.DataFrame()

#import all the trading data
total = pd.DataFrame()
for info in os.listdir(data_dir):
    domain = os.path.abspath(data_dir)
    info = os.path.join(domain,info)
    info = pd.read_csv(info)
    total = pd.concat([info,total])

    #for size and sign we just use every data point to do it.

#get all the dates in this set

dates = set(list(total['date']))
#generate a random date-time series for random sampling
t = []
for x in range(9,15):
    for y in range(0,60):
        for z in range(0,60):
            x = str(x)
            y = str(y)
            z = str(z)
            if len(str(x)) == 1:
                x = '0'+x
            if len(str(y)) == 1:
                y = '0' + y
            if len(str(z)) == 1:
                z = '0' + z
            t.append(x + y + z)
ta = t[1500:9001][:-300]+t[14400:][:-299]

tb = t[1500:9001][300:]+t[14400:][300:]
ta1 = t[1500:9001][:-60]+t[14400:][:-59]
tb1 = t[1500:9001][60:]+t[14400:][60:]
T = t[1500:9001]+t[14400:]
tb.append('150000')
tb1.append('150000')
time5after = pd.DataFrame(tb,index=ta)
time1after = pd.DataFrame(tb1,index=ta1)
td = []
for i in ta:
    for j in dates:
        td.append(str(j)+i)
td1 = []
for i in ta1:
    for j in dates:
        td1.append(str(j)+i)
#Method 1: MLE Method
def poi(k,lda):
    y = lda**k*np.e**(-lda)/np.math.factorial(k)
    return y
def bino(x,p):
    if x == 1:
        return p
    if x == -1:
        return 1-p
def expo(lda,x):
    p = lda*np.e**(-lda*x)
    return p


    
#MLE for \lambda
k_list = []
ft = pd.Series(td)
fts = ft.sample(1000000)
for i in fts.index:
    date = np.int64(fts[i][:8])
    t_5 = int(time5after.loc[fts[i][-6:]]+'000')
    time = np.int64((fts[i][-6:]+'000'))
    sp = total.loc[(total['date']==date)]
    sp = sp.loc[sp['time']>=time]
    sp = sp.loc[sp['time']<=t_5]
    k = len(sp)
    k_list.append(k)
#Calibrate \lambda using method of moments
print(np.mean(k_list))
 

def logL_poi(lda):
    L = -np.log(np.sum(np.array([k*np.log(lda)+np.log(np.e)*(-lda)-np.sum([np.log(x) for x in range(1,k+1)]) for k in k_list])))
    return L
#res = minimize(logL_poi,np.array([100,200,300,400,800]), method='nelder-mead')
#dist1 = st.poisson
#res = dist1.fit(k_list,floc = 0)

#MLE for \beta of size

v_list = list(total['ntrade'])#We can change the number of sample by will.
def logL_expo(lda):
    L = -np.log(np.prod(np.array([expo(lda,x) for x in v_list])))
    return L
#res = minimize(logL_expo,0.5) #this do not work well

dist2 = st.expon#Define the distribution function
loc, beta = dist2.fit(v_list,floc = 0)#MLE fit

#Method of Moments to get \beta
b = np.mean(v_list)
print(b)
print(beta)
#MLE to estimate p

#Method of moments to get p
bs = total.sample(2000)['BS']



#Calculate autocorrelation rho and p

bs_series = total['BS'].loc[total['BS']!=' ']
bs_series = bs_series.replace('B',1)
bs_series = bs_series.replace('S',-1)
rho = bs_series.autocorr()

p = (np.mean(bs_series)+1)/2
#Soulution for Q3
#for SH600519
data_dir2 = 'E:\\Desktop\\2019_Spring\\Microstructure\\SH600519_quote'


#import all the trading data
total_quote = pd.DataFrame()
for info in os.listdir(data_dir2):
    domain = os.path.abspath(data_dir2)
    info = os.path.join(domain,info)
    info = pd.read_csv(info)
    total_quote = pd.concat([info,total_quote])
total_quote = total_quote.loc[total_quote['price']!=0]
total_quote['mid_quote'] = (total_quote['BidPrice1']+total_quote['AskPrice1'])/2
q = total_quote.loc[total_quote['mid_quote']>total_quote['price']] 
q = q.loc[q['BS']=='S']
uo = total_quote.loc[total_quote['mid_quote']<total_quote['price']] 
uo = uo.loc[uo['BS']=='B']
te = total_quote.loc[total_quote['mid_quote']==total_quote['price']] 
te = te.loc[te['BS']==' ']
qt = pd.concat([q,uo])
acc_519 = len(qt)/len(total_quote)
#for SH601398
data_dir3 = 'E:\\Desktop\\2019_Spring\\Microstructure\\SH601398_quote'
dataset = pd.DataFrame()

#import all the trading data
total_quote1 = pd.DataFrame()
for info in os.listdir(data_dir3):
    domain = os.path.abspath(data_dir3)
    info = os.path.join(domain,info)
    info = pd.read_csv(info)
    total_quote1 = pd.concat([info,total_quote1])
total_quote1 = total_quote1.loc[total_quote1['price']!=0]
total_quote1['mid_quote'] = (total_quote1['BidPrice1']+total_quote1['AskPrice1'])/2
q = total_quote1.loc[total_quote1['mid_quote']>total_quote1['price']] 
q = q.loc[q['BS']=='S']
uo = total_quote1.loc[total_quote1['mid_quote']<total_quote1['price']] 
uo = uo.loc[uo['BS']=='B']
te = total_quote1.loc[total_quote1['mid_quote']==total_quote1['price']] 
te = te.loc[te['BS']==' ']
qt1 = pd.concat([q,uo])
acc_398 = len(qt1)/len(total_quote1)
t_acc = (len(qt1)+len(qt))/(len(total_quote1)+len(total_quote))

#Q4 Solution
data_dir2 = 'E:\\Desktop\\2019_Spring\\Microstructure\\SH600519_quote'
total_quote = pd.DataFrame()
for info in os.listdir(data_dir2):
    domain = os.path.abspath(data_dir2)
    info = os.path.join(domain,info)
    info = pd.read_csv(info)
    total_quote = pd.concat([info,total_quote])
    
total_quote['mid_quote'] = (total_quote['BidPrice1']*total_quote['AskVolume1']+total_quote['AskPrice1']*total_quote['BidVolume1'])/(total_quote['BidVolume1']+total_quote['AskVolume1'])
total_quote = total_quote.loc[total_quote['price']!=0]
total_quote1['mid_quote'] = (total_quote1['BidPrice1']*total_quote1['AskVolume1']+total_quote1['AskPrice1']*total_quote1['BidVolume1'])/(total_quote1['BidVolume1']+total_quote1['AskVolume1'])
total_quote1 = total_quote1.loc[total_quote1['price']!=0]
V_bar1 = np.mean(total_quote['volume'])
V_bar2 = np.mean(total_quote1['volume'])
total_quote = total_quote.replace('B',1)
total_quote = total_quote.replace('S',-1)
total_quote1 = total_quote1.replace('B',1)
total_quote1 = total_quote1.replace('S',-1)
total_quote = total_quote.loc[total_quote['BS']!=' ']
total_quote1 = total_quote1.loc[total_quote1['BS']!=' ']
total_quote = total_quote.loc[total_quote['BidPrice1']!=0]
total_quote1 = total_quote1.loc[total_quote1['BidPrice1']!=0]
total_quote = total_quote.loc[total_quote['AskPrice1']!=0]
total_quote1 = total_quote1.loc[total_quote1['AskPrice1']!=0]
#sample the time of 5 min and 1 min


ib1 = pd.DataFrame()
ib2 = pd.DataFrame()
ft1 = pd.Series(td)
ft2 = pd.Series(td1)
fts1 = ft1.sample(10000)
fts2 = ft2.sample(10000)
IMB_listx = []
Ret_listx = []
IMB_listy = []
Ret_listy = []
# 5 min 600519
for i in fts1.index:
    date = np.int64(fts1[i][:8])
    t_5 = int(time5after.loc[fts1[i][-6:]])
    time = np.int64((fts1[i][-6:]))
    sp = total_quote.loc[(total_quote['date']==date)]
    sp = sp.loc[sp['time']>=time]
    sp = sp.loc[sp['time']<=t_5]
    if len(sp) > 0:
        Ret = (float(sp[-1:]['mid_quote'])-float(sp[:1]['mid_quote']))/float(sp[:1]['mid_quote'])
        IMB = sum(sp['BS']*sp['volume'])/V_bar1
        if IMB > 0:
            IMB_listx.append(IMB)
            Ret_listx.append(Ret)
        if IMB < 0:
            IMB_listy.append(IMB)
            Ret_listy.append(Ret)
x = np.array(IMB_listx)
y = np.array(Ret_listx)
def func(x,beta,gam):
    f = np.std(x)*beta*np.abs(x)**gam
    return f
popt, pcov = curve_fit(func, x, y)
a = popt[0] 
b = popt[1]
print(a)
print(b)
yvals = func(x,a,b)
plot1 = plt.plot(x, y, 's',label='original values')
plot2 = plt.plot(x, yvals, 'r',label='polyfit values')
plt.title('5 min 600519 fit model when IMB>0')
plt.show()
plt.close()
x = np.array(IMB_listy)
y = np.array(Ret_listy)
popt, pcov = curve_fit(func, x, y)
a = popt[0] 
b = popt[1]
print(a)
print(b)
yvals = func(x,a,b)
plot1 = plt.plot(x, y, 's',label='original values')
plot2 = plt.plot(x, yvals, 'r',label='polyfit values')
plt.title('5 min 600519 fit model when IMB<0')
plt.show()
plt.close()
# 1 min 600519
IMB_list1 = []
Ret_list1 = []
IMB_list2 = []
Ret_list2 = []
for i in fts2.index:
    date = np.int64(fts2[i][:8])
    t_5 = int(time1after.loc[fts2[i][-6:]])
    time = np.int64((fts2[i][-6:]))
    sp = total_quote.loc[(total_quote['date']==date)]
    sp = sp.loc[sp['time']>=time]
    sp = sp.loc[sp['time']<=t_5]
    if len(sp) > 0:
        Ret = (float(sp[-1:]['mid_quote'])-float(sp[:1]['mid_quote']))/float(sp[:1]['mid_quote'])
        IMB = sum(sp['BS']*sp['volume'])/V_bar1
        if IMB > 0:
            IMB_list1.append(IMB)
            Ret_list1.append(Ret)
        if IMB < 0:
            IMB_list2.append(IMB)
            Ret_list2.append(Ret)
x = np.array(IMB_list2)
y = np.array(Ret_list2)
popt, pcov = curve_fit(func, x, y)
a = popt[0] 
b = popt[1]
print(a)
print(b)
yvals = func(x,a,b)
plot1 = plt.plot(x, y, 's',label='original values')
plot2 = plt.plot(x, yvals, 'r',label='polyfit values')
plt.title('1 min 600519 fit model when IMB<0')
plt.show()
plt.close()
x = np.array(IMB_list1)
y = np.array(Ret_list1)
popt, pcov = curve_fit(func, x, y)
a = popt[0] 
b = popt[1]
print(a)
print(b)
yvals = func(x,a,b)
plot1 = plt.plot(x, y, 's',label='original values')
plot2 = plt.plot(x, yvals, 'r',label='polyfit values')
plt.title('1 min 600519 fit model when IMB>0')
plt.show()
plt.close()
# 5 min 601398
IMB_listx = []
IMB_listy = []
Ret_listx = []
Ret_listy = []
for i in fts1.index:
    date = np.int64(fts1[i][:8])
    t_5 = int(time5after.loc[fts1[i][-6:]])
    time = np.int64((fts1[i][-6:]))
    sp = total_quote1.loc[(total_quote1['date']==date)]
    sp = sp.loc[sp['time']>=time]
    sp = sp.loc[sp['time']<=t_5]
    if len(sp) > 0:
        Ret = (float(sp[-1:]['mid_quote'])-float(sp[:1]['mid_quote']))/float(sp[:1]['mid_quote'])
        IMB = sum(sp['BS']*sp['volume'])/V_bar2
        if IMB > 0:
            IMB_listx.append(IMB)
            Ret_listx.append(Ret)
        if IMB < 0:
            IMB_listy.append(IMB)
            Ret_listy.append(Ret)
x = np.array(IMB_listx)
y = np.array(Ret_listx)
popt, pcov = curve_fit(func, x, y)
a = popt[0] 
b = popt[1]
print(a)
print(b)
yvals = func(x,a,b)
plot1 = plt.plot(x, y, 's',label='original values')
plot2 = plt.plot(x, yvals, 'r',label='polyfit values')
plt.title('5 min 601398 fit model when IMB>0')
plt.show()
plt.close()
x = np.array(IMB_listy)
y = np.array(Ret_listy)
popt, pcov = curve_fit(func, x, y)
a = popt[0] 
b = popt[1]
print(a)
print(b)
yvals = func(x,a,b)
plot1 = plt.plot(x, y, 's',label='original values')
plot2 = plt.plot(x, yvals, 'r',label='polyfit values')
plt.title('5 min 601398 fit model when IMB<0')
plt.show()
plt.close()
# 1 min 601398
IMB_list1 = []
Ret_list1 = []
IMB_list2 = []
Ret_list2 = []
for i in fts2.index:
    date = np.int64(fts2[i][:8])
    t_5 = int(time1after.loc[fts2[i][-6:]])
    time = np.int64((fts2[i][-6:]))
    sp = total_quote1.loc[(total_quote1['date']==date)]
    sp = sp.loc[sp['time']>=time]
    sp = sp.loc[sp['time']<=t_5]
    if len(sp) > 0:
        Ret = (float(sp[-1:]['mid_quote'])-float(sp[:1]['mid_quote']))/float(sp[:1]['mid_quote'])
        IMB = sum(sp['BS']*sp['volume'])/V_bar2
        if IMB > 0:
            IMB_list1.append(IMB)
            Ret_list1.append(Ret)
        if IMB < 0:
            IMB_list2.append(IMB)
            Ret_list2.append(Ret)
x = np.array(IMB_list1)
y = np.array(Ret_list1)
popt, pcov = curve_fit(func, x, y)
a = popt[0] 
b = popt[1]
print(a)
print(b)
yvals = func(x,a,b)
plot1 = plt.plot(x, y, 's',label='original values')
plot2 = plt.plot(x, yvals, 'r',label='polyfit values')
plt.title('1 min 601398 fit model when IMB>0')
plt.show()
plt.close()
x = np.array(IMB_list2)
y = np.array(Ret_list2)
popt, pcov = curve_fit(func, x, y)
a = popt[0] 
b = popt[1]
print(a)
print(b)
yvals = func(x,a,b)
plot1 = plt.plot(x, y, 's',label='original values')
plot2 = plt.plot(x, yvals, 'r',label='polyfit values')
plt.title('1 min 601398 fit model when IMB<0')
plt.show()
plt.close()