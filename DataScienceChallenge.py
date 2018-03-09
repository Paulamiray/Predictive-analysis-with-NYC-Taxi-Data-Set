
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots, show
from scipy.stats import lognorm, skew, chisquare,ttest_ind,f_oneway
get_ipython().run_line_magic('matplotlib', 'inline')

# Execute this block for Question 1 : Start
# Load the dataset
data = pd.read_csv('green_tripdata_2015-09.csv')

# Print the size of the dataset
print ("Number of rows:", data.shape[0])
print ("Number of columns: ", data.shape[1])
# Execute this block for Question 1 : End 

# Execute this block for Question 2 : Start
# Defining the subplots for the figure
# Source:https://matplotlib.org/api/_as_gen/matplotlib.pyplot.subplots.html
f,ax = plt.subplots(1,2,figsize = (20,5)) 

# Histogram of the number of trip distance
data.Trip_distance.hist(bins=30,ax=ax[0])
ax[0].set_xlabel('Trip Distance(in miles)')
ax[0].set_ylabel('Number of Occurence')
ax[0].set_yscale('log')
ax[0].set_title('Histogram of Trip Distance with outliers')

# Define Vector for Trip Distance
v = data.Trip_distance 
# Plot Histogram with 30 bins and not considering the points outside 3 standard deviations
v[~((v-v.median()).abs()>3*v.std())].hist(bins=30,ax=ax[1]) 
ax[1].set_xlabel('Trip Distance(in miles)')
ax[1].set_ylabel('Number of Occurence')
ax[1].set_title('Histogram of Trip Distance')

# Using lognormal fit for the distribution.
# Source: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.lognorm.html
# Using the mean of trip distance as the scale parameter
scatter,loc,mean = lognorm.fit(data.Trip_distance.values,scale=data.Trip_distance.mean(),loc=0)
pdf_fitted = lognorm.pdf(np.arange(0,12,.1),scatter,loc,mean)
ax[1].plot(np.arange(0,12,.1),600000*pdf_fitted,'r') 
ax[1].legend(['Data','Lognormal distribution fit'])
plt.show()
# Execute this block for Question 2 : End

# Execute this block for Question 3 (1st part) : Start
# Changing the date format for pickup and drop off locations.
data['Pickup_dt'] = data.lpep_pickup_datetime.apply(lambda x:dt.datetime.strptime(x,"%Y-%m-%d %H:%M:%S"))
data['Dropoff_dt'] = data.Lpep_dropoff_datetime.apply(lambda x:dt.datetime.strptime(x,"%Y-%m-%d %H:%M:%S"))

# Create a new variable for pickup hours
data['Pickup_hour'] = data.Pickup_dt.apply(lambda x:x.hour)

# Mean and Median of trip distance by pickup hour
# Creating a pivot table to aggregate Trip_distance by hour
table1 = data.pivot_table(index='Pickup_hour', values='Trip_distance',aggfunc=('mean','median')).reset_index()
# Rename columns
table1.columns = ['Hour','Mean_distance','Median_distance']
print ('*** Trip distance by hour of the day ***\n')
print(table1)

# Plot to show the Mean trip distance per hour
data[['Trip_distance','Pickup_hour']].groupby('Pickup_hour').mean().plot.bar()
plt.title('Mean trip distance across the day')
plt.show()

#Plot to show the Median trip distance per hour
data[['Trip_distance','Pickup_hour']].groupby('Pickup_hour').median().plot.bar()
plt.title('Median trip distance across the day')
plt.show()
# Execute this block for Question 3 (1st part) : End

# Execute this block for Question 3 (2nd part) : Start
# Airport trips
airport_trips = data[(data.RateCodeID==2) | (data.RateCodeID==3)]
nonairport_trips = data[(data.RateCodeID==1) | (data.RateCodeID==4)| (data.RateCodeID==5)| (data.RateCodeID==6)]
print ("Total number of trips to/from NYC airports: ", airport_trips.shape[0])
# Print out the average fare as shown in the meter to/from NYC airports
print ("Average Metered fare to/from NYC airports : $", airport_trips.Fare_amount.mean(),"per trip")
# Print out the average total fare after tax to/from NYC airports
print ("Average Total fare of trips to/from NYC airports : $", airport_trips.Total_amount.mean(),"per trip")

# Non-Airport trips
print ("Total number of other trips : ", nonairport_trips.shape[0])
print ("Average Metered fare for other non-airport trips : $", nonairport_trips.Fare_amount.mean(),"per trip")

# Vector for Trip Distance of
# airport trips
v1 = airport_trips.Trip_distance 
# non-airport trips
v2 = data.loc[~data.index.isin(v1.index),'Trip_distance'] 

# To remove outliers I have excluded any data point that is located further than 3 standard deviations 
v1 = v1[~((v1-v1.median()).abs()>3*v1.std())]
v2 = v2[~((v2-v2.median()).abs()>3*v2.std())] 

# Define bins boundaries
bins = np.histogram(v1,normed=True)[1]
h1 = np.histogram(v1,bins=bins,normed=True)
h2 = np.histogram(v2,bins=bins,normed=True)

# Plot distributions of trip distance normalized among groups
f,ax = plt.subplots(1,2,figsize = (20,5))
w = .4*(bins[1]-bins[0])
ax[0].bar(bins[:-1],h1[0],alpha=1,width=w,color='b')
ax[0].bar(bins[:-1]+w,h2[0],alpha=1,width=w,color='g')
ax[0].legend(['Airport trips','Non-airport trips'],loc='best',title='')
ax[0].set_xlabel('Trip distance (miles)')
ax[0].set_ylabel('Trip Count (grouped)')
ax[0].set_title('Histogram of Trip distance distribution')
# Execute this block for Question 3 (2nd part) : End

# Execute this block for Question 5 2nd part: Start
# Plot hourly distribution
airport_trips.Pickup_hour.value_counts(normalize=True).sort_index().plot(ax=ax[1])
data.loc[~data.index.isin(v2.index),'Pickup_hour'].value_counts(normalize=True).sort_index().plot(ax=ax[1])
ax[1].set_xlabel('Hours')
ax[1].set_ylabel('Trips count')
ax[1].set_title('Hourly distribution of trips')
ax[1].legend(['Airport trips','Non-airport trips'],loc='best',title='')
plt.show()
# Execute this block for Question 5 2nd part: End

# Execute this block for Question 4 (1st part) : Start
#Derived variable Tip_percentage
data = data[(data.Total_amount>=2.5)] #cleaning
data['Tip_percentage'] = 100*data.Tip_amount/data.Total_amount
print((data['Tip_percentage']>10).value_counts())
print ("Summary: Tip percentage\n",data.Tip_percentage.describe())
# Execute this block for Question 4 (1st part) : End 

#Restart the kernel and execute the below code
# Feature Engineering Begins
import pandas as pd
import numpy as np
import datetime as dt
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots, show
from scipy.stats import lognorm, skew, chisquare,f_oneway,ttest_ind
get_ipython().run_line_magic('matplotlib', 'inline')

# Load the dataset
data = pd.read_csv('green_tripdata_2015-09.csv')

def feature_eng(adata):
    
    # Copy of the original dataset
    data = adata.copy()
   # Changing the date format for pickup and drop off locations.
    data['Pickup_dt'] = data.lpep_pickup_datetime.apply(lambda x:dt.datetime.strptime(x,"%Y-%m-%d %H:%M:%S"))
    data['Dropoff_dt'] = data.Lpep_dropoff_datetime.apply(lambda x:dt.datetime.strptime(x,"%Y-%m-%d %H:%M:%S"))

    # Create a new variable for pickup hours
    data['Pickup_hour'] = data.Pickup_dt.apply(lambda x:x.hour)
    # derive time variables
    print ("Time variables")
    ref_week = dt.datetime(2015,9,1).isocalendar()[1] # first week of september in 2015
    data['Week'] = data.Pickup_dt.apply(lambda x:x.isocalendar()[1])-ref_week+1
    data['Week_day']  = data.Pickup_dt.apply(lambda x:x.isocalendar()[2])
    data['Month_day'] = data.Pickup_dt.apply(lambda x:x.day)
    data['Hour'] = data.Pickup_dt.apply(lambda x:x.hour)
    
    # Trip duration 
    print ("Trip_duration")
    data['Trip_duration'] = ((data.Dropoff_dt-data.Pickup_dt).apply(lambda x:x.total_seconds()/60.))
    
    print ("Direction variables")
    # create direction variable Direction_NS. 
    # This is 2 if taxi moving from north to south, 1 in the opposite direction and 0 otherwise
    data['Direction_NS'] = (data.Pickup_latitude>data.Dropoff_latitude)*1+1
    indices = data[(data.Pickup_latitude == data.Dropoff_latitude) & (data.Pickup_latitude!=0)].index
    data.loc[indices,'Direction_NS'] = 0

    # create direction variable Direction_EW. 
    # This is 2 if taxi moving from east to west, 1 in the opposite direction and 0 otherwise
    data['Direction_EW'] = (data.Pickup_longitude>data.Dropoff_longitude)*1+1
    indices = data[(data.Pickup_longitude == data.Dropoff_longitude) & (data.Pickup_longitude!=0)].index
    data.loc[indices,'Direction_EW'] = 0
    
    # create variable for Speed
    print ("Deriving Speed. Make sure to check for possible NaNs and Inf vals")
    data['Speed_mph'] = data.Trip_distance/(data.Trip_duration/60)
    # replace all NaNs values and values >240mph by a values sampled from a random distribution of 
    # mean 12.9 and  standard deviation 6.8mph. These values were extracted from the distribution
    indices_oi = data[(data.Speed_mph.isnull()) | (data.Speed_mph>240)].index
    data.loc[indices_oi,'Speed_mph'] = np.abs(np.random.normal(loc=12.9,scale=6.8,size=len(indices_oi)))
    print ("Feature engineering successfully done!")
    
    # create tip percentage variable
    data['Tip_percentage'] = 100*data.Tip_amount/data.Total_amount
    
    # create with_tip variable
    data['With_tip'] = (data.Tip_percentage>0)*1

    return data

# run the code in new cell to create new features on the dataset
print ("size before feature engineering:", data.shape)
data = feature_eng(data)
print ("size after feature engineering:", data.shape)
# Feature Engineering Ends

# Data Cleaning Process Begins
def data_clean(adata):
    ## Make a copy of the input
    data = adata.copy()
    ## Drop Ehail_fee: 99% of its values are NaNs
    if 'Ehail_fee' in data.columns:
        data.drop('Ehail_fee',axis=1,inplace=True)

    ## Replace missing values in Trip_type with the most frequent value 1
    data['Trip_type '] = data['Trip_type '].replace(np.NaN,1)
  
    ## Remove negative values from Total amound and Fare_amount
    print ("Negative values found and replaced by their abs")
    print ("Total_amount", 100*data[data.Total_amount<0].shape[0]/float(data.shape[0]),"%")
    print ("Fare_amount", 100*data[data.Fare_amount<0].shape[0]/float(data.shape[0]),"%")
    print ("Improvement_surcharge", 100*data[data.improvement_surcharge<0].shape[0]/float(data.shape[0]),"%")
    print ("Tip_amount", 100*data[data.Tip_amount<0].shape[0]/float(data.shape[0]),"%")
    print ("Tolls_amount", 100*data[data.Tolls_amount<0].shape[0]/float(data.shape[0]),"%")
    print ("MTA_tax", 100*data[data.MTA_tax<0].shape[0]/float(data.shape[0]),"%")
    data.Total_amount = data.Total_amount.abs()
    data.Fare_amount = data.Fare_amount.abs()
    data.improvement_surcharge = data.improvement_surcharge.abs()
    data.Tip_amount = data.Tip_amount.abs()
    data.Tolls_amount = data.Tolls_amount.abs()
    data.MTA_tax = data.MTA_tax.abs()
    
    # RateCodeID
    indices_oi = data[~((data.RateCodeID>=1) & (data.RateCodeID<=6))].index
    data.loc[indices_oi, 'RateCodeID'] = 2 # 2 = Cash payment was identified as the common method
    print (round(100*len(indices_oi)/float(data.shape[0]),2),"% of values in RateCodeID were invalid.Replaced by the most frequent 2")
    
    # Extra
    indices_oi = data[~((data.Extra==0) | (data.Extra==0.5) | (data.Extra==1))].index
    data.loc[indices_oi, 'Extra'] = 0 # 0 was identified as the most frequent value
    print (round(100*len(indices_oi)/float(data.shape[0]),2),"% of values in Extra were invalid.Replaced by the most frequent 0")
    
    # Total_amount: the minimum charge is 2.5, so I will replace every thing less than 2.5 by the median 11.76 (pre-obtained in analysis)
    indices_oi = data[(data.Total_amount<2.5)].index
    data.loc[indices_oi,'Total_amount'] = 11.76
    print (round(100*len(indices_oi)/float(data.shape[0]),2),"% of values in total amount worth <$2.5.Replaced by the median 1.76")
    
    # encode categorical to numeric (I avoid to use dummy to keep dataset small)
    if data.Store_and_fwd_flag.dtype.name != 'int64':
        data['Store_and_fwd_flag'] = (data.Store_and_fwd_flag=='Y')*1
    
    # rename time stamp variables and convert them to the right format
    print ("renaming variables...")
    data.rename(columns={'lpep_pickup_datetime':'Pickup_dt','Lpep_dropoff_datetime':'Dropoff_dt'},inplace=True)
    print ("converting timestamps variables to right format ...")
   # data['Pickup_dt'] = data.Pickup_dt.apply(lambda x:dt.datetime.strptime(x,"%Y-%m-%d %H:%M:%S"))
   # data['Dropoff_dt'] = data.Dropoff_dt.apply(lambda x:dt.datetime.strptime(x,"%Y-%m-%d %H:%M:%S"))
   
    print ("Done cleaning")
    return data

# Run code in new cell to clean the data
data = data_clean(data)
# Data Cleaning Process Ends

# Explanatory analysis Begins
## Comparing the two Tip_percentage identified groups
# Spliting data in the two groups
data1 = data[data.Tip_percentage>0]
data2 = data[data.Tip_percentage==0]

# Histograms to compare
fig,ax=plt.subplots(1,2,figsize=(14,4))
data.Tip_percentage.hist(bins = 20,normed=True,ax=ax[0])
ax[0].set_xlabel('Percentage of Tip')
ax[0].set_ylabel('Group normed count')
ax[0].set_title('Histogram for Tip Percent for all transactions')

data1.Tip_percentage.hist(bins = 20,normed=True,ax=ax[1])
ax[1].set_xlabel('Percentage of Tip')
ax[1].set_ylabel('Group normed count')
ax[1].set_title('Histogram for Distribution of Transactions with tips')


plt.show()

# Functions for exploratory data analysis
# Define function to visualize continous variables
def visualize_continuous(df,label,method={'type':'histogram','bins':20},outlier='on'):
    # Create vector 
    v = df[label]
    # Mean and standard deviation
    m = v.mean()
    s = v.std()
    
    f,ax = plt.subplots(1,2,figsize=(14,4))
    ax[0].set_title('Distribution of '+label)
    ax[1].set_title('Tip % by '+label)
    
    if outlier=='off': 
        v = v[(v-m)<=3*s]
        ax[0].set_title('Distribution of '+label+'(no outliers)')
        ax[1].set_title('Tip % by '+label+'(no outliers)')
    if method['type'] == 'histogram': # plot the histogram
        v.hist(bins = method['bins'],ax=ax[0])
    if method['type'] == 'boxplot': # plot the box plot
        df.loc[v.index].boxplot(label,ax=ax[0])
    ax[1].plot(v,df.loc[v.index].Tip_percentage,'.',alpha=0.4)
    ax[0].set_xlabel(label)
    ax[1].set_xlabel(label)
    ax[0].set_ylabel('Count')
    ax[1].set_ylabel('Tip (%)')
    
# Define function to visualize categorical variables
def visualize_categories(df,catName,chart_type='histogram',ylimit=[None,None]):
    print (catName)
    cats = sorted(pd.unique(df[catName]))
    if chart_type == 'boxplot': # boxplot
        generate_boxplot(df,catName,ylimit)
    elif chart_type == 'histogram': # histogram
        generate_histogram(df,catName)
    else:
        pass
        
    groups = df[[catName,'Tip_percentage']].groupby(catName).groups 
    tips = df.Tip_percentage
    if len(cats)<=2: # if there are only two groups use t-test
        print (ttest_ind(tips[groups[cats[0]]],tips[groups[cats[1]]]))
    else: # otherwise, use one_way anova test
        cmd = "f_oneway("
    for cat in cats:
        cmd+="tips(groups["+str(cat)+"]),"
        cmd=cmd[:-1]+")"
        print ("one way anova test:", eval(cmd)) #evaluate the command and print
    print ("Frequency of categories (%):\n",df[catName].value_counts(normalize=True)*100)

# Define function to test groups with_tip and without_tip are different at 95% of confidence level
def test_classification(df,label,yl=[0,50]):
    if len(pd.unique(df[label]))==2: #Check if the variable is categorical and run chisquare test
        vals=pd.unique(df[label])
        gp1 = df[df.With_tip==0][label].value_counts().sort_index()
        gp2 = df[df.With_tip==1][label].value_counts().sort_index()
        print ("t-test if", label, "can be used to distinguish transaction with tip and without tip")
        print (chisquare(gp1,gp2))
    elif len(pd.unique(df[label]))>=10: # Else run the t-test
        df.boxplot(label,by='With_tip')
        plt.ylim(yl)
        plt.show()
        print ("t-test if", label, "can be used to distinguish transaction with tip and without tip")
        print ("results:",ttest_ind(df[df.With_tip==0][label].values,df[df.With_tip==1][label].values,False))
    else:
        pass
    
# Define Boxplot
def generate_boxplot(df,catName,ylimit):
    df.boxplot('Tip_percentage',by=catName)
    #plt.title('Tip % by '+catName)
    plt.title('')
    plt.ylabel('Tip (%)')
    if ylimit != [None,None]:
        plt.ylim(ylimit)
    plt.show()
    
# Define Histogram
def generate_histogram(df,catName):
    cats = sorted(pd.unique(df[catName]))
    colors = plt.cm.jet(np.linspace(0,1,len(cats)))
    hx = np.array(map(lambda x:round(x,1),np.histogram(df.Tip_percentage,bins=20)[1]))
    fig,ax = plt.subplots(1,1,figsize = (15,4))
    for i,cat in enumerate(cats):
        vals = df[df[catName] == cat].Tip_percentage
        h = np.histogram(vals,bins=hx)
        w = 0.9*(hx[1]-hx[0])/float(len(cats))
        plt.bar(hx[:-1]+w*i,h[0],color=colors[i],width=w)
    plt.legend(cats)
    plt.yscale('log')
    plt.title('Distribution of Tip by '+catName)
    plt.xlabel('Tip (%)')
    
    
# visualization of the Payment_type
visualize_categories(data1,'Payment_type','boxplot',[13,20])

# Example of exploration of the Fare_amount using the implented code:
visualize_continuous(data1,'Fare_amount',outlier='on')
test_classification(data,'Fare_amount',[0,25])

# Explanatory analysis section Ends

# Execute this block for Question 5 1st part: Start
from scipy.stats import lognorm, skew, chisquare,ttest_ind,f_oneway
# calculate anova
hours = range(24)
cmd = "f_oneway("
for h in hours:
    cmd+="data[data.Hour=="+str(h)+"].Speed_mph,"
cmd=cmd[:-1]+")"
print ("one way anova test:", eval(cmd)) #evaluate the command and print

# boxplot
data.boxplot('Speed_mph','Hour')
plt.ylim([5,24]) # cut off outliers
plt.ylabel('Speed (mph)')
plt.show()
# Execute this block for Question 5 (1st part) : End

# Execute this block for Question 4 (2nd part) : Start
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import mean_squared_error
from sklearn import cross_validation, metrics   
from sklearn.grid_search import GridSearchCV   

# Function to train models and perform cross validation.
#This is used in both classification and regression
def modelfit(alg,dtrain,predictors,target,scoring_method,performCV=True,printFeatureImportance=True,cv_folds=5):
       # Train the algorithm 
    alg.fit(dtrain[predictors],dtrain[target])
       # Predict on train set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    if scoring_method == 'roc_auc':
        dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
    
    # Perform cross-validation
    if performCV:
        cv_score = cross_validation.cross_val_score(alg,dtrain[predictors],dtrain[target],cv=cv_folds,scoring=scoring_method)
        # Print model report
        print ("\nModel report:")
        if scoring_method == 'roc_auc':
            print ("Accuracy:",metrics.accuracy_score(dtrain[target].values,dtrain_predictions))
            print ("AUC Score (Train):",metrics.roc_auc_score(dtrain[target], dtrain_predprob))
        if (scoring_method == 'mean_squared_error'):
            print ("Accuracy:",metrics.mean_squared_error(dtrain[target].values,dtrain_predictions))
    if performCV:
        print ("CV Score - Mean : %.7g | Std : %.7g | Min : %.7g | Max : %.7g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))
    # Print feature importances
    if printFeatureImportance:
        if dir(alg)[0] == '_Booster': #runs only if alg is xgboost
            feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
        else:
            feat_imp = pd.Series(alg.feature_importances_,predictors).sort_values(ascending=False)
        feat_imp.plot(kind='bar',title='Feature Importance')
        plt.ylabel('Feature Import Score')
        plt.show()

# Optimize n_estimator through grid search.
#This functions is used to tune paremeters of a predictive algorithm
def optimize_num_trees(alg,param_test,scoring_method,train,predictors,target):
    gsearch = GridSearchCV(estimator=alg, param_grid = param_test, scoring=scoring_method,n_jobs=2,iid=False,cv=5)
    gsearch.fit(train[predictors],train[target])
    return gsearch

# Plot optimization results
def plot_opt_results(alg):
    cv_results = []
    for i in range(len(param_test['n_estimators'])):
        cv_results.append((alg.grid_scores_[i][1],alg.grid_scores_[i][0]['n_estimators']))
    cv_results = pd.DataFrame(cv_results)
    plt.plot(cv_results[1],cv_results[0])
    plt.xlabel('# trees')
    plt.ylabel('score')
    plt.title('optimization report')
    
#####################################################################################################
from sklearn.ensemble import GradientBoostingClassifier
import os, json, requests, pickle
from sklearn.preprocessing import normalize, scale
print ("Optimizing the classifier...")

# Plot optimization results
def plot_opt_results(alg):
    cv_results = []
    for i in range(len(param_test['n_estimators'])):
        cv_results.append((alg.grid_scores_[i][1],alg.grid_scores_[i][0]['n_estimators']))
    cv_results = pd.DataFrame(cv_results)
    plt.plot(cv_results[1],cv_results[0])
    plt.xlabel('# trees')
    plt.ylabel('score')
    plt.title('optimization report')
    
# Copy of the training set
train = data.copy() 
# Sample sie taken as 100000 , 5 fold CV done
train = train.loc[np.random.choice(train.index,size=100000,replace=False)]
target = 'With_tip' 

# Initiate the timing
tic = dt.datetime.now() 

predictors = ['Payment_type','Total_amount','Trip_duration','Speed_mph','MTA_tax',
              'Extra','Hour','Direction_NS', 'Direction_EW']

# Optimize n_estimator through grid search
param_test = {'n_estimators': list(range(30,151,20))} # define range over which number of trees is to be optimized

# Initiate classification model
model_cls = GradientBoostingClassifier(
    learning_rate=0.1, 
    min_samples_split=2,
    max_depth=5,
    max_features='auto',
    subsample=0.8, 
    random_state = 10)

# Get results of the search grid
gs_cls = optimize_num_trees(model_cls,param_test,'roc_auc',train,predictors,target)
print (gs_cls.grid_scores_, gs_cls.best_params_, gs_cls.best_score_)

# Cross validate the best model with optimized number of estimators
modelfit(gs_cls.best_estimator_,train,predictors,target,'roc_auc')

# Save the best estimator on disk as pickle for a later use
# Source: https://pythonprogramming.net/python-pickle-module-save-objects-serialization/
with open('my_classifier.pkl','wb') as fid:
    pickle.dump(gs_cls.best_estimator_,fid)
    fid.close()
    
print ("Processing time:", dt.datetime.now()-tic)

###################################################

# Testing on a different set
indices = data.index[~data.index.isin(train.index)]
test = data.loc[np.random.choice(indices,size=100000,replace=False)]

ypred = gs_cls.best_estimator_.predict(test[predictors])

print ("ROC AUC:", metrics.roc_auc_score(ypred,test.With_tip))
###############################################################

import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split

train = data1.copy()
train = train.loc[np.random.choice(train.index,size=100000,replace=False)]
indices = data1.index[~data1.index.isin(train.index)]
test = data1.loc[np.random.choice(indices,size=100000,replace=False)]

train['ID'] = train.index
IDCol = 'ID'
target = 'Tip_percentage'

predictors = ['VendorID', 'Passenger_count', 'Trip_distance', 'Total_amount', 
              'Extra', 'MTA_tax', 'Tolls_amount', 'Payment_type', 
              'Hour', 'Week', 'Week_day', 'Month_day', 'Shift_type', 
              'Direction_NS', 'Direction_EW', 'Trip_duration', 'Speed_mph']
predictors = ['Trip_distance','Tolls_amount', 'Direction_NS', 'Direction_EW', 'Trip_duration', 'Speed_mph']
predictors = ['Total_amount', 'Trip_duration', 'Speed_mph']


# Random Forest 
# Source: https://www.codeproject.com/Articles/1197167/Random-Forest-Python
tic = dt.datetime.now()
from sklearn.ensemble import RandomForestRegressor

# Optimize n_estimator through grid search
param_test = {'n_estimators': list(range(50,200,25))} # define range over which number of trees is to be optimized

# Initiate classification model
rfr = RandomForestRegressor()#n_estimators=100)
# Get results of the search grid
gs_rfr = optimize_num_trees(rfr,param_test,'neg_mean_squared_error',train,predictors,target)

# print optimization results
print (gs_rfr.grid_scores_, gs_rfr.best_params_, gs_rfr.best_score_)

# Cross validate the best model with optimized number of estimators
modelfit(gs_rfr.best_estimator_,train,predictors,target,'neg_mean_squared_error')

# Save the best estimator on disk as pickle for a later use
# Source: https://pythonprogramming.net/python-pickle-module-save-objects-serialization/
with open('my_rfr_reg2.pkl','wb') as fid:
    pickle.dump(gs_rfr.best_estimator_,fid)
    fid.close()

ypred = gs_rfr.best_estimator_.predict(test[predictors])

print ('RFR test mse:',metrics.neg_mean_squared_error(ypred,test.Tip_percentage))
print ('RFR r2:', metrics.r2_score(ypred,test.Tip_percentage))
print (dt.datetime.now()-tic)
plot_opt_results(gs_rfr)

#########################################################################################
# Predicts the percentage tip expected 
def predict_tip(transaction):
    # define predictors labels as per optimization results
    cls_predictors = ['Payment_type','Total_amount','Trip_duration','Speed_mph','MTA_tax',
                      'Extra','Hour','Direction_NS', 'Direction_EW']
    reg_predictors = ['Total_amount', 'Trip_duration', 'Speed_mph']
    
    # Classify transactions
    clas = gs_cls.best_estimator_.predict(transaction[cls_predictors])
    
    # Predict tips for those transactions classified as 1
    return clas*gs_rfr.best_estimator_.predict(transaction[reg_predictors])
#######################################################################################
# Make Final Prediction
test = data.loc[np.random.choice(data.index,size = 100000,replace=False)]
ypred = predict_tip(test)
print ("final mean_squared_error:", metrics.mean_squared_error(ypred,test.Tip_percentage))
print ("final r2_score:", metrics.r2_score(ypred,test.Tip_percentage))

# Execute this block for Question 4 (2nd part) : End

