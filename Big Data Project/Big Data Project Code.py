import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
#%% Data loading
data = pd.read_csv('middleSchoolData.csv')
data2 = pd.read_csv('middleSchoolData.csv')

#drop rows with nans
public_schools = data.iloc[:485]
public_schools = public_schools.dropna()
public_schools = public_schools.reset_index(drop=True)
#449 public schools left after dropping rows with missing vals

charter_schools = data.iloc[485:]
charter_schools = charter_schools.dropna(subset = ['dbn', 'school_name', 'applications', 'acceptances',
        'asian_percent',
       'black_percent', 'hispanic_percent', 'multiple_percent',
       'white_percent', 'rigorous_instruction', 'collaborative_teachers',
       'supportive_environment', 'effective_school_leadership',
       'strong_family_community_ties', 'trust', 'disability_percent',
       'poverty_percent', 'ESL_percent', 'school_size', 'student_achievement',
       'reading_scores_exceed', 'math_scores_exceed'])
charter_schools = charter_schools.reset_index(drop=True)
#65 charter schools left after dropping rows with nans besides avg class size and per student spending
data = pd.concat([public_schools, charter_schools])
data = data.reset_index(drop=True)
#514 total obs left

#%%
print(data.columns)
#%% Some Data Wrangling
app_rate = data['applications'][:]/data['school_size'][:]

plt.hist(data['applications'])
plt.xlabel('applications')
plt.show()

plt.hist(app_rate)
plt.xlabel('application rate')
plt.show()
# both applications and application rates are extremely right skewedj

modded_apps = data['applications'].to_numpy()
for i in range(len(modded_apps)):
    if modded_apps[i] == 0:
        modded_apps[i] = 1

acceptance_rate = data['acceptances'].to_numpy()/modded_apps
#%%
print(np.max(acceptance_rate))
#%% Question 1
#What is the correlation between the number of applications and admissions to HSPHS?

#correlation between no. of applications and acceptances
"""
app_admit_corr = np.corrcoef(data['applications'][:],data['acceptances'][:])
print("correlation between apps and admissions is:", app_admit_corr[0,1])
"""

pearson_corrs = data['applications'].corr(data['acceptances'], method='pearson')
print("pearson correlation between apps and admissions is:", 
      pearson_corrs) #0.806
# since it isn't very linear, let's try a spearman correlation
spearman_corrs = data['applications'].corr(data['acceptances'], method='spearman')
print("spearman correlation between apps and admissions is:", 
      spearman_corrs) #0.942

plt.plot(data['applications'][:],data['acceptances'][:],'o')
plt.xlabel('applications')
plt.ylabel('acceptances')
#%% Question 2
#What is a better predictor of admission to HSPHS? Raw number of applications or
#application *rate*?

#correlation between application rates and acceptances
"""
app_rate = data['applications'][:]/data['school_size'][:]
app_rate_admit_corr = np.corrcoef(app_rate, data['acceptances'][:])
print("correlation between app rate and admission is:", app_rate_admit_corr[0,1])
"""
pearson_corrs = app_rate.corr(data['acceptances'], method='pearson')
print("pearson correlation between app rate and admission is:", 
      pearson_corrs) #0.683
spearman_corrs = app_rate.corr(data['acceptances'], method='spearman')
print("spearman correlation between app rate and admission is:", 
      spearman_corrs) #0.778

plt.plot(app_rate, data['acceptances'][:], 'o')
plt.xlabel('application rate')
plt.ylabel('acceptances')

#%% Question 3
#Which school has the best *per student* odds of sending someone to HSPHS?

#calculate the admittance rates for all the schools
#defined as acceptances/school size
admit_rate = data['acceptances'][:]/data['school_size'][:]
#checking if the index is correct
print(np.max(admit_rate)) #0.235
print(admit_rate.at[274])
#get index of school with best admit rate
print('index of max admit rate',admit_rate.idxmax()) #303
#print name of school
print(data.at[274,'school_name']) #THE CHRISTA MCAULIFFE SCHOOL\I.S. 187

#calculate the acceptance rates for all the schools
#defined as acceptances/applications
accept_rate = data['acceptances'][:]/data['applications'][:]

print('index of max accept rate', accept_rate.idxmax())
print(accept_rate.at[274]) #0.817

#we might be interested in two things here: the proportion of students 
#in the entire school that get into HSPHS 
#and the proportion of students that get in out of the ones that applied.



#%% Question 4
#Is there a relationship between how students perceive their school (as reported in columns
#L-Q) and how the school performs on objective measures of achievement (as noted in
#columns V-X)

print(data.iloc[:,[11,12,13,14,15,16,21,22,23]].columns)
corr_matrix = data.iloc[:,[11,12,13,14,15,16,21,22,23]].corr(method='pearson')

plt.imshow(corr_matrix)
plt.colorbar()
#We can see that the perceptions are largely correlated with each other and
#student achievement and scores are correlated with each other


#%% PCA for school climate and student achievement
data_arr = data.iloc[:,[11,12,13,14,15,16,21,22,23]].to_numpy()
zscoredData = stats.zscore(data_arr)
pca = PCA()
pca.fit(zscoredData)
eigValues = pca.explained_variance_ 
loadings = pca.components_
rotatedData = pca.fit_transform(zscoredData)


numPredictors = 9
plt.bar(np.linspace(1,numPredictors,numPredictors),eigValues)
plt.title('Scree plot')
plt.xlabel('Principal Components')
plt.ylabel('Eigenvalues')
plt.show()

PC = 0
plt.bar(np.linspace(1,9,9),loadings[PC,:])
plt.xlabel('Columns')
plt.ylabel('Loading')
plt.show()

#basically, PC1 is school climate and PC2 is measures of achievement
plt.plot(rotatedData[:449,0],rotatedData[:449,1],'o')  #blue for public
plt.plot(rotatedData[449:,0],rotatedData[449:,1],'ro') #red for charter
plt.xlabel('School Climate')
plt.ylabel('Achievement')
plt.show()
#correlation basically 0 for all schools but 0.625 for charter schools
#print(np.corrcoef(rotatedData[449:,0],rotatedData[449:,1]))


#%% Question 5
#Test a hypothesis of your choice
#Difference between charter schools and non-charter schools in achievement
#We will use our principal component for measures of achievement
#There are 524 observations left after dropping rows with non-systematically missing data
#The last 65 obs are charter schools
#The first 459 are non-charter
PC = 1
non_charter = rotatedData[:449,PC]
charter = rotatedData[449:,PC]
mean_charter = np.mean(charter)
mean_non_charter = np.mean(non_charter)
print('mean of charter is', mean_charter)
print('mean of non-charter is', mean_non_charter)
t, p = stats.ttest_ind(charter, non_charter)
print(t,p) #t statistic of: 3.577, p-value of: 0.00038
#At a 1% significance level we reject the null hypothesis 
#that there is no difference in student achievement 
#between charter and non-charter schools and 
#conclude that charter schools perform better on measures of achievement
"""
print()
median_charter = np.median(charter)
median_non_charter = np.median(non_charter)
print('median of charter is', median_charter)
print('median of non-charter is', median_non_charter)
U, p = stats.mannwhitneyu(charter, non_charter)
print(U,p)
"""
#%% Question 6
#Is there any evidence that the availability of material resources (e.g. per student spending
#or class size) impacts objective measures of achievement or admission to HSPHS?

#We can only analyze public schools for this question
#print(public_schools['avg_class_size'].corr(public_schools['acceptances'],method='pearson'))
#0.36
plt.plot(public_schools['avg_class_size'],public_schools['acceptances'],'o')
plt.xlabel('Avg Class Size')
plt.ylabel('Acceptances')
plt.show()

#print(public_schools['per_pupil_spending'].corr(public_schools['acceptances'],method='pearson'))
#-0.34
plt.plot(public_schools['per_pupil_spending'],public_schools['acceptances'],'o')
plt.xlabel('Per Student Spending')
plt.ylabel('Acceptances')
plt.show()

#%%

#print(np.corrcoef(public_schools['avg_class_size'].to_numpy(),rotatedData[:449,1]))
#0.21
#0.41 with principal component
plt.plot(public_schools['avg_class_size'],rotatedData[:449,1],'o')
plt.xlabel('Avg Class Size')
plt.ylabel('Student Achievement (scores)')
m, b = np.polyfit(public_schools['avg_class_size'],rotatedData[:449,1],1)
plt.plot(public_schools['avg_class_size'], m*public_schools['avg_class_size']+b, color='red')
plt.show()
#R^2 is 0.168

#print(np.corrcoef(public_schools['per_pupil_spending'].to_numpy(),rotatedData[:449,1]))
#-0.15
#-0.42 with principal component
plt.plot(public_schools['per_pupil_spending'],rotatedData[:449,1],'o')
plt.xlabel('Per Student Spending')
plt.ylabel('Student Achievement (scores)')
m, b = np.polyfit(public_schools['per_pupil_spending'],rotatedData[:449,1],1)
plt.plot(public_schools['per_pupil_spending'], m*public_schools['per_pupil_spending']+b, color='red')
plt.show()
#R^2 is 0.177


#%% Question 7
#What proportion of schools accounts for 90% of all students accepted to HSPHS? 

acceptances = data['acceptances'].to_numpy()
#4282 acceptances total
#90% is 3853.8 ~ 3854

sorted_accept = np.flip(np.sort(acceptances))

plt.hist(acceptances, bins = 40)
plt.xlabel = 'Acceptances'

cutoff_sum = 0
counter = 0
for i in sorted_accept:
    if cutoff_sum <= 3854:
        cutoff_sum += i
        counter += 1
print(counter)
#103 out of 514 schools (20%) account for 90% of acceptances

#%% Question 8
#Build a model of your choice – clustering, classification or prediction – that includes all
#factors – as to what school characteristics are most important in terms of a) sending
#students to HSPHS, b) achieving high scores on objective measures of achievement?

def rmse(predictions, targets):
    differences = predictions - targets                       #the DIFFERENCEs.
    differences_squared = differences ** 2                    #the SQUAREs of ^
    mean_of_differences_squared = differences_squared.mean()  #the MEAN of ^
    rmse_val = np.sqrt(mean_of_differences_squared)           #ROOT of ^
    return rmse_val    
   

treeModel = DecisionTreeRegressor()

#We will not include avg class size and per student spending for model for all schools

climate_component = pd.Series(rotatedData[:,0])
achievement_component = pd.Series(rotatedData[:,1])


#%%
#first model for predicting acceptances in all schools
X = data[['applications','white_percent','hispanic_percent','black_percent','asian_percent','multiple_percent',
          'disability_percent','poverty_percent','ESL_percent','school_size']].copy()
X['climate_component'] = climate_component.values
X['achievement_component'] = achievement_component.values
y = data['acceptances'].copy()
np.random.seed(0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

treeModel.fit(X_train, y_train)
predicted_y = treeModel.predict(X_test)
print('rmse is', rmse(predicted_y, y_test)) # 8.52
plt.barh(X.columns, treeModel.feature_importances_)
plt.xlabel('Feature importances')
#applications are the most important which is not surprising
#besides applications, it seems the most important features are
#hispanic percent, multiple percent, ESL percent, and achievement component

#%%
#second model for predicting student achievement
X = data[['applications','white_percent','hispanic_percent','black_percent','asian_percent','multiple_percent',
          'acceptances','disability_percent','poverty_percent','ESL_percent','school_size']].copy()
X['climate_component'] = climate_component.values
y = pd.Series(achievement_component, name='achievement_component')
np.random.seed(42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

treeModel.fit(X_train, y_train)
predicted_y = treeModel.predict(X_test)
print('rmse is', rmse(predicted_y, y_test)) #1.24
plt.barh(X.columns, treeModel.feature_importances_)
plt.xlabel('Feature importances')
#most important features are poverty percent, disability percent,
#climate component, and school size.

#%%
#third model out of curiosity for what predicts applications per student
X = data[['white_percent','hispanic_percent','black_percent','asian_percent','multiple_percent',
          'disability_percent','poverty_percent','ESL_percent']].copy()
X['climate_component'] = climate_component.values
X['achievement_component'] = achievement_component.values
y = data['applications'].copy()
np.random.seed(101)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

treeModel.fit(X_train, y_train)
predicted_y = treeModel.predict(X_test)
print('rmse is', rmse(predicted_y, y_test)) # 0.07

plt.barh(X.columns, treeModel.feature_importances_)


#%%
#fourth mode for acceptance rates
X = data[['white_percent','hispanic_percent','black_percent','asian_percent','multiple_percent',
          'disability_percent','poverty_percent','ESL_percent','school_size']].copy()
X['climate_component'] = climate_component.values
X['achievement_component'] = achievement_component.values
y = data['acceptances'][:]/data['applications'][:].copy()
np.random.seed(1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

treeModel.fit(X_train, y_train)
predicted_y = treeModel.predict(X_test)
print('rmse is', rmse(predicted_y, y_test)) # 0.12
plt.barh(X.columns, treeModel.feature_importances_)
plt.xlabel('Feature importances')

#achievement and school climate are the most important











