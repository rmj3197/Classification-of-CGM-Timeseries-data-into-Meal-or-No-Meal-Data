#!/usr/bin/env python
# coding: utf-8

# # Reading the files (CHANGE THE NAME OF THE FILE HERE)

# In[36]:


import csv
f = open('your file.csv')
csv_f = csv.reader(f)

data = []

for row in csv_f:
    data.append(row)


# # Defining all the feature functions

# In[37]:


import pandas as pd
import csv
import numpy as np
from collections import Counter 
import warnings
warnings.filterwarnings('ignore')
import io


# In[38]:


from scipy.fftpack import fft, ifft
def fft_and_peaks(df):
        
    fft_val=[]
    fft_val.append(abs(fft(df)))
        
    peak_val=[]
    for z in range(len(fft_val)):
        a = list(set(fft_val[z]))
        a.sort()
        a = a[::-1][1:5]
        peak_val.append(a)
    return(fft_val,peak_val)


# In[39]:


def autocorrelation(df):
        
    autocorr_lag_2=[]
    autocorr_lag_3=[]
    autocorr_lag_4=[]
    autocorr_lag_5=[]
    autocorr_lag_6=[]
    autocorr_lag_7=[]
    autocorr_lag_8=[]
    autocorr_lag_9=[]
    autocorr_lag_10=[]
    autocorr_lag_11=[]
    autocorr_lag_12=[]
    autocorr_lag_13=[]
    autocorr_lag_14=[]

    auto_corr_2= df.autocorr(lag=2)
    autocorr_lag_2.append(auto_corr_2)

    auto_corr_3= df.autocorr(lag=3)
    autocorr_lag_3.append(auto_corr_3)

    auto_corr_4= df.autocorr(lag=4)
    autocorr_lag_4.append(auto_corr_4)

    auto_corr_5= df.autocorr(lag=5)
    autocorr_lag_5.append(auto_corr_5)

    auto_corr_6= df.autocorr(lag=6)
    autocorr_lag_6.append(auto_corr_6)

    auto_corr_7= df.autocorr(lag=7)
    autocorr_lag_7.append(auto_corr_7)

    auto_corr_8= df.autocorr(lag=8)
    autocorr_lag_8.append(auto_corr_8)

    auto_corr_9= df.autocorr(lag=9)
    autocorr_lag_9.append(auto_corr_9)

    auto_corr_10=df.autocorr(lag=10)
    autocorr_lag_10.append(auto_corr_10)

    auto_corr_11= df.autocorr(lag=11)
    autocorr_lag_11.append(auto_corr_11)

    auto_corr_12= df.autocorr(lag=12)
    autocorr_lag_12.append(auto_corr_12)

    auto_corr_13= df.autocorr(lag=13)
    autocorr_lag_13.append(auto_corr_13)

    auto_corr_14= df.autocorr(lag=14)
    autocorr_lag_14.append(auto_corr_14)

    df1 = list(zip(autocorr_lag_2, autocorr_lag_3,autocorr_lag_4,autocorr_lag_5,autocorr_lag_6,autocorr_lag_7,autocorr_lag_8,autocorr_lag_9,autocorr_lag_10,autocorr_lag_11,autocorr_lag_12,autocorr_lag_13,autocorr_lag_14))[0]
    return (df1)


# In[40]:


def polyfit_coeffs(df):
    coeff_list = []
    y = df.to_numpy()
    time = np.linspace(1,df.shape[0],df.shape[0])
    coeff = np.polyfit(time,y,6)
    coeff_list.append(coeff)
    df1 = coeff_list[0]
    return df1


# In[41]:


def velocity(data):
    
    data = data.reset_index().drop(columns = 'index').T
    mean = []
    std =[]
    median =[]
    
    interval = 10
    for k in range(len(data)):
        window_size = 5
        velocity = []
        row_data = data.iloc[k].values
        row_length = len(row_data)
        counter = 0
        cgmvel = []
        for i in range((len(row_data) - window_size)):
            cgmvel.append(counter)
            counter += 5
            p = (row_data[i] - row_data[i + window_size])
            vel = p / interval
            velocity.append(vel)
        mean.append(np.mean(velocity))
        std.append(np.std(velocity))
        median.append(np.median(velocity))
        
    df = list(zip(mean, std, median))[0]
    return df


# In[42]:


def psd(df):
    import scipy.signal as signal
    max_amp=[]
    std_amp=[]
    mean_amp=[]
    
    df = df.reset_index().drop(columns = 'index').T
    
    for k in range(len(df)):
        v, welch_values  = np.array((signal.welch(df.iloc[k].values)))
        max_amp.append(np.sqrt(max(welch_values)))
        std_amp.append(np.std(np.sqrt(welch_values)))
        mean_amp.append(np.mean(np.sqrt(welch_values)))
        
    df1 = list(zip(max_amp, std_amp, mean_amp))[0]
    return df1
   


# # Calculating Features

# In[43]:


fft_peak_val =[]
cgm_vel = []
auto_corr=[]
polyfit_coeff=[]
psd_val = []


for i in range(len(data)):
    intermediate = pd.DataFrame(data[i]).reset_index().drop(columns = 'index')
    each_row = pd.to_numeric(intermediate[0],errors='coerce').interpolate().dropna()
    
    
    if (len(each_row)!=0):
        x,y = fft_and_peaks(each_row)
        fft_peak_val.append(y[0])

        v = velocity(each_row)
        cgm_vel.append(v)

        acr = autocorrelation(each_row)
        auto_corr.append(acr)

        p = polyfit_coeffs(each_row)
        polyfit_coeff.append(p)

        psd1 = psd(each_row)
        psd_val.append(psd1)
    else:
        pass


# # Generating the Dataset

# In[44]:


peak_val = pd.DataFrame(list(fft_peak_val),columns = ['Peak 2','Peak 3','Peak 4','Peak 5'])
cgm_vel_val = pd.DataFrame(cgm_vel,columns=['vel_Mean','vel_STD','vel_Median'])
autocorr_val = pd.DataFrame(auto_corr,columns=['autocorr_lag_2', 'autocorr_lag_3','autocorr_lag_4','autocorr_lag_5','autocorr_lag_6','autocorr_lag_7','autocorr_lag_8','autocorr_lag_9','autocorr_lag_10','autocorr_lag_11','autocorr_lag_12','autocorr_lag_13','autocorr_lag_14']) 
polyfit_val = pd.DataFrame(polyfit_coeff,columns=['coeff1','coeff2','coeff3','coeff4','coeff5','coeff6','coeff7'])
psd_val = pd.DataFrame(psd_val,columns =['max amplitude', 'std amplitude','mean amplitude'])


# In[45]:


dataset = pd.concat([peak_val,cgm_vel_val,autocorr_val,polyfit_val,psd_val],axis=1).fillna(0)


# # PCA

# In[46]:


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy import stats
feature_matrix = StandardScaler().fit_transform(dataset)
pca = PCA(n_components=5)
principalComponents = pca.fit_transform(feature_matrix)
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2','principal component 3','principal component 4','principal component 5'])


# # Prediction using Pre-Trained Pickle file

# In[47]:


from joblib import dump, load
with open('RandomForestClassifier.pickle', 'rb') as pre_trained:
    pickle_file = load(pre_trained)
    predict = pickle_file.predict(principalDf)    
    pre_trained.close()
    print('Prediction - ')
    print (predict)

