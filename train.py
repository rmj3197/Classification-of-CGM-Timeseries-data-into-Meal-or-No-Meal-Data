#!/usr/bin/env python
# coding: utf-8


# In[179]:


import pandas as pd
import csv
import numpy as np
from collections import Counter 
import warnings
warnings.filterwarnings('ignore')
import io


# In[180]:


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


# In[181]:


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


# In[182]:


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


# In[183]:


def polyfit_coeffs(df):
    coeff_list = []
    y = df.to_numpy()
    time = np.linspace(1,df.shape[0],df.shape[0])
    coeff = np.polyfit(time,y,6)
    coeff_list.append(coeff)
    df1 = coeff_list[0]
    return df1


# In[184]:


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
   


# In[185]:


def file_read(filename):
    f = open(filename)
    csv_f = csv.reader(f)

    data = []

    for row in csv_f:
        data.append(row)
    return data


# In[186]:


def calculate_features(data):   
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
    
    peak_val = pd.DataFrame(list(fft_peak_val),columns = ['Peak 2','Peak 3','Peak 4','Peak 5'])
    cgm_vel_val = pd.DataFrame(cgm_vel,columns=['vel_Mean','vel_STD','vel_Median'])
    autocorr_val = pd.DataFrame(auto_corr,columns=['autocorr_lag_2', 'autocorr_lag_3','autocorr_lag_4','autocorr_lag_5','autocorr_lag_6','autocorr_lag_7','autocorr_lag_8','autocorr_lag_9','autocorr_lag_10','autocorr_lag_11','autocorr_lag_12','autocorr_lag_13','autocorr_lag_14']) 
    polyfit_val = pd.DataFrame(polyfit_coeff,columns=['coeff1','coeff2','coeff3','coeff4','coeff5','coeff6','coeff7'])
    psd_val = pd.DataFrame(psd_val,columns =['max amplitude', 'std amplitude','mean amplitude'])
    
    dataset = pd.concat([peak_val,cgm_vel_val,autocorr_val,polyfit_val,psd_val],axis=1).fillna(0)
    return(dataset)


# # Reading Meal Data and Calculating the features

# In[187]:


meal_pat1 = file_read('mealData1.csv')
features_meal_pat1 = calculate_features(meal_pat1)


# In[188]:


meal_pat2 = file_read('mealData2.csv')
features_meal_pat2 = calculate_features(meal_pat2)


# In[189]:


meal_pat3 = file_read('mealData3.csv')
features_meal_pat3 = calculate_features(meal_pat3)


# In[190]:


meal_pat4 = file_read('mealData4.csv')
features_meal_pat4 = calculate_features(meal_pat4)


# In[191]:


meal_pat5 = file_read('mealData5.csv')
features_meal_pat5 = calculate_features(meal_pat5)


# In[192]:


meal_data = pd.concat([features_meal_pat1,features_meal_pat2,features_meal_pat3,features_meal_pat4,features_meal_pat5],axis=0)


# In[193]:


meal_data['Label']=1


# # Reading Non-Meal Data and Calculating the features

# In[194]:


nomeal_pat1 = file_read('Nomeal1.csv')
features_nomeal_pat1 = calculate_features(nomeal_pat1)


# In[195]:


nomeal_pat2 = file_read('Nomeal2.csv')
features_nomeal_pat2 = calculate_features(nomeal_pat2)


# In[196]:


nomeal_pat3 = file_read('Nomeal3.csv')
features_nomeal_pat3 = calculate_features(nomeal_pat3)


# In[197]:


nomeal_pat4 = file_read('Nomeal4.csv')
features_nomeal_pat4= calculate_features(nomeal_pat2)


# In[198]:


nomeal_pat5 = file_read('Nomeal5.csv')
features_nomeal_pat5 = calculate_features(nomeal_pat5)


# In[199]:


nomeal_data = pd.concat([features_nomeal_pat1,features_nomeal_pat2,features_nomeal_pat3,features_nomeal_pat4,features_nomeal_pat5],axis=0)


# In[200]:


nomeal_data['Label']=0


# # Combining Meal and No Meal Data

# In[201]:


from sklearn.utils import shuffle
dataset = shuffle(pd.concat([meal_data,nomeal_data],axis=0).fillna(0)).reset_index().drop(columns = ['index'])


# # PCA

# In[202]:


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy import stats
feature_matrix = StandardScaler().fit_transform(dataset.drop(columns= ['Label']))
pca = PCA(n_components=5)
principalComponents = pca.fit_transform(feature_matrix)
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2','principal component 3','principal component 4','principal component 5'])


# # K-Fold Cross Validation

# In[205]:


principalDf['Label'] = dataset['Label']


# In[206]:


principalDfData = principalDf.drop(columns = ['Label'])


# In[207]:


def get_score(model, X_train,X_test,y_train,y_test):
    model.fit(X_train,y_train)
    return model.score(X_test,y_test)


# In[208]:


from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from xgboost import XGBClassifier
scores_rf = []

from sklearn.model_selection import KFold
kfold = KFold(n_splits=10,shuffle=False)
for train_index, test_index in kfold.split(principalDfData):
    X_train,X_test,y_train,y_test = principalDfData.loc[train_index],principalDfData.loc[test_index],    principalDf.Label.loc[train_index],principalDf.Label.loc[test_index]
    scores_rf.append(get_score(RandomForestClassifier(criterion="entropy"), X_train, X_test, y_train, y_test))


# In[209]:


print(np.mean(scores_rf))


# # Pickling the Model

# In[210]:


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
X, y= principalDfData, principalDf['Label']
clf.fit(X,y)


# In[211]:


from joblib import dump, load
dump(clf, 'RandomForestClassifier.pickle')


# In[ ]:




