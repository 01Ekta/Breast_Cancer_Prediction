#!/usr/bin/env python
# coding: utf-8

# In[3]:


get_ipython().system('pip install xgboost')


# In[4]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np 
import pandas as pd
import seaborn as s
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
#optimum parameter choosing 
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from xgboost import XGBClassifier
import pickle
import os 
import warnings
warnings.filterwarnings('ignore')


# In[5]:


data = pd.read_csv('data.csv')
data


# In[6]:


data.shape


# In[7]:


df = data


# In[8]:


df['diagnosis'].value_counts()


# In[9]:


df.dtypes


# In[10]:


df['diagnosis']=df['diagnosis'].astype('category')
df.dtypes


# In[11]:


df.head()


# In[12]:


x= df.drop (labels='diagnosis' ,axis =1 )
x


# In[13]:


y = df['diagnosis']
y


# In[14]:


col = x. columns
col


# In[15]:


x.isnull().sum()


# In[16]:


df_norm = (x- x.mean()) / (x.max()- x.min())
df_norm= pd.concat ([df_norm,y], axis =1 )
df_norm


# In[17]:


df.drop('diagnosis',axis =1).drop('id',axis =1).corr()


# In[18]:


plt.rcParams['figure.figsize']=(20,12)
s.set(font_scale=1.4)
s.heatmap (df.drop('diagnosis',axis =1).drop('id',axis =1).corr(),cmap = 'coolwarm',annot = True)


# In[19]:


plt.rcParams['figure.figsize']=(20,8)
f, (ax1,ax2,ax3,ax4,ax5) = plt.subplots (1,5)
s.boxplot ('diagnosis', y = 'radius_mean',data = df , ax = ax1)
s.boxplot ('diagnosis', y = 'texture_mean',data = df , ax = ax2)
s.boxplot ('diagnosis', y = 'perimeter_mean',data = df , ax = ax3)
s.boxplot ('diagnosis', y = 'area_mean',data = df , ax = ax4)
s.boxplot ('diagnosis', y = 'smoothness_mean',data = df , ax = ax5)
f .tight_layout()

f, (ax1,ax2,ax3,ax4,ax5) = plt.subplots (1,5)
s.boxplot ('diagnosis', y = 'compactness_mean',data = df , ax = ax1)
s.boxplot ('diagnosis', y = 'concavity_mean',data = df , ax = ax2)
s.boxplot ('diagnosis', y = 'concave points_mean',data = df , ax = ax3)
s.boxplot ('diagnosis', y = 'symmetry_mean',data = df , ax = ax4)
s.boxplot ('diagnosis', y = 'fractal_dimension_mean',data = df , ax = ax5)
f .tight_layout()


# In[20]:


g = s.FacetGrid (df,col = 'diagnosis', hue = 'diagnosis')
g.map (s.distplot, "radius_mean", hist = False, rug = True)

g = s.FacetGrid (df,col = 'diagnosis', hue = 'diagnosis')
g.map (s.distplot, 'texture_mean', hist = False, rug = True)

g = s.FacetGrid (df,col = 'diagnosis', hue = 'diagnosis')
g.map (s.distplot, 'perimeter_mean', hist = False, rug = True)

g = s.FacetGrid (df,col = 'diagnosis', hue = 'diagnosis')
g.map (s.distplot, "area_mean", hist = False, rug = True)

g = s.FacetGrid (df,col = 'diagnosis', hue = 'diagnosis')
g.map (s.distplot, "smoothness_mean", hist = False, rug = True)

g = s.FacetGrid (df,col = 'diagnosis', hue = 'diagnosis')
g.map (s.distplot, "compactness_mean", hist = False, rug = True)

g = s.FacetGrid (df,col = 'diagnosis', hue = 'diagnosis')
g.map (s.distplot, "concavity_mean", hist = False, rug = True)

g = s.FacetGrid (df,col = 'diagnosis', hue = 'diagnosis')
g.map (s.distplot, "concave points_mean", hist = False, rug = True)

g = s.FacetGrid (df,col = 'diagnosis', hue = 'diagnosis')
g.map (s.distplot, "symmetry_mean", hist = False, rug = True)

g = s.FacetGrid (df,col = 'diagnosis', hue = 'diagnosis')
g.map (s.distplot, "fractal_dimension_mean", hist = False, rug = True)


# In[21]:


plt.rcParams['figure.figsize']=(20,8)
f, (ax1,ax2,ax3,ax4,ax5) = plt.subplots (1,5)
s.boxplot ('diagnosis', y = 'radius_se',data = df , ax = ax1,palette = 'cubehelix')
s.boxplot ('diagnosis', y = 'texture_se',data = df , ax = ax2,palette = 'cubehelix')
s.boxplot ('diagnosis', y = 'perimeter_se',data = df , ax = ax3,palette = 'cubehelix')
s.boxplot ('diagnosis', y = 'area_se',data = df , ax = ax4,palette = 'cubehelix')
s.boxplot ('diagnosis', y = 'smoothness_se',data = df , ax = ax5,palette = 'cubehelix')
f .tight_layout()

f, (ax1,ax2,ax3,ax4,ax5) = plt.subplots (1,5)
s.boxplot ('diagnosis', y = 'compactness_se',data = df , ax = ax1,palette = 'cubehelix')
s.boxplot ('diagnosis', y = 'concavity_se',data = df , ax = ax2,palette = 'cubehelix')
s.boxplot ('diagnosis', y = 'concave points_se',data = df , ax = ax3,palette = 'cubehelix')
s.boxplot ('diagnosis', y = 'symmetry_se',data = df , ax = ax4,palette = 'cubehelix')
s.boxplot ('diagnosis', y = 'fractal_dimension_se',data = df , ax = ax5,palette = 'cubehelix')
f .tight_layout()


# In[22]:


plt.rcParams['figure.figsize']=(20,8)
f, (ax1,ax2,ax3,ax4,ax5) = plt.subplots (1,5)
s.boxplot ('diagnosis', y = 'radius_worst',data = df , ax = ax1,palette = 'coolwarm')
s.boxplot ('diagnosis', y = 'texture_worst',data = df , ax = ax2,palette = 'coolwarm')
s.boxplot ('diagnosis', y = 'perimeter_worst',data = df , ax = ax3,palette = 'coolwarm')
s.boxplot ('diagnosis', y = 'area_worst',data = df , ax = ax4,palette = 'coolwarm')
s.boxplot ('diagnosis', y = 'smoothness_worst',data = df , ax = ax5,palette = 'coolwarm')
f .tight_layout()

f, (ax1,ax2,ax3,ax4,ax5) = plt.subplots (1,5)
s.boxplot ('diagnosis', y = 'compactness_worst',data = df , ax = ax1,palette = 'coolwarm')
s.boxplot ('diagnosis', y = 'concavity_worst',data = df , ax = ax2,palette = 'coolwarm')
s.boxplot ('diagnosis', y = 'concave points_worst',data = df , ax = ax3,palette = 'coolwarm')
s.boxplot ('diagnosis', y = 'symmetry_worst',data = df , ax = ax4,palette = 'coolwarm')
s.boxplot ('diagnosis', y = 'fractal_dimension_worst',data = df , ax = ax5,palette = 'coolwarm')
f .tight_layout()


# In[23]:


x_norm = df_norm.drop (labels= 'diagnosis', axis =1 )
y_norm = df_norm ['diagnosis']
col = x_norm. columns
print (col)
display (x_norm)
display (y_norm)


# In[24]:


le = LabelEncoder()
le.fit (y_norm)
y_norm = le.transform(y_norm)
y_norm = pd.DataFrame(y_norm)
print (y_norm)


# In[25]:


def FitModel (X,Y, algo_name , algorithm, gridSearchParams, cv):
    np.random.seed(10)
    x_train, x_test, y_train, y_test = train_test_split (X,Y,test_size = 0.2)
    
    grid = GridSearchCV(estimator = algorithm, param_grid = gridSearchParams,
                        cv = cv, scoring = 'accuracy', verbose = 1 , n_jobs = -1 )
    
    grid_result = grid.fit(x_train, y_train)
    best_params = grid_result.best_params_
    pred = grid_result.predict (x_test)
    cm = confusion_matrix (y_test,pred)
    
    print (pred)
    pickle.dump(grid_result,open(algo_name,'wb'))
    
    print ('Best Params :', best_params)
    print ('Classification Report:',classification_report(y_test,pred))
    print ('Accuracy Score', (accuracy_score(y_test,pred)))
    print ('Confusion Matrix :\n',cm)


# In[26]:


param = {
            'C': [0.1,1,100,1000],
            'gamma':[0.0001,0.001, 0.005, 0.1,1, 3,5,10, 100]
         }

FitModel (x_norm,y_norm,'SVC',SVC(), param, cv =10)


# In[27]:


param = { 'n_estimators': [100,500,1000,2000]  }
FitModel (x_norm,y_norm,'Random Forest',RandomForestClassifier(), param, cv =10)


# In[28]:


np.random.seed(10)
x_train,x_test, y_train,y_test = train_test_split (x_norm,y_norm,test_size = 0.2)
forest = RandomForestClassifier (n_estimators = 500)
fit = forest.fit (x_train, y_train)
accuracy = fit.score(x_test,y_test)
predict = fit.predict(x_test)
cmatrix = confusion_matrix (y_test, predict)
print ('Classification Report:',classification_report(y_test,predict))
print ('Accuracy Score', (accuracy_score(y_test,predict)))
print ('Accuracy of Random Forest ', (accuracy))
print ('Confusion Matrix :\n',cmatrix)


# In[29]:


param = { 'n_estimators': [100,500,1000,2000]  }
FitModel (x_norm,y_norm,'XGBoost', XGBClassifier(),param, cv = 10)


# In[30]:


importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]

print ("Feature Ranking:")
for f in range (x.shape[1]):
    print ("Feature %s (%f)"  %(list (x)[f],importances[indices[f]]))


# In[31]:


feat_imp = pd.DataFrame({'Feature': list(x), 'Gini importance': importances[indices]})
plt.rcParams['figure.figsize']= (12,12)
s.set_style ('whitegrid')
ax= s.barplot(x ='Gini importance', y = 'Feature', data = feat_imp  )
ax.set (xlabel = 'Gini Importances')
plt.show()


# In[34]:


get_ipython().system('pip install imblearn')


# In[35]:


from imblearn.over_sampling import SMOTE


# In[36]:


df['diagnosis'].value_counts()


# In[37]:


sm = SMOTE(random_state =42)
X_res, Y_res = sm.fit_resample (x_norm, y_norm)


# In[38]:


Y_res[0].value_counts()


# In[39]:


param = { 'n_estimators': [100,500,1000,2000]  }
FitModel (X_res, Y_res ,'Random Forest',RandomForestClassifier(), param, cv =10)


# In[40]:


param = {
            'C': [0.1,1,100,1000],
            'gamma':[0.0001,0.001, 0.005, 0.1,1, 3,5,10, 100]
         }
FitModel (X_res, Y_res,'SVC',SVC(), param, cv =10)


# In[41]:


param = { 'n_estimators': [100,500,1000,2000]  }
FitModel (X_res, Y_res,'XGBoost', XGBClassifier(),param, cv = 10)


# In[42]:


feat_imp.index = feat_imp.Feature


# In[43]:


feat_to_keep = feat_imp.iloc[1:15].index
type(feat_to_keep),feat_to_keep


# In[44]:


X_res = pd.DataFrame(X_res)
Y_res = pd.DataFrame(Y_res)
X_res.columns = x_norm.columns
param = { 'n_estimators': [100,500,1000,2000]  }
FitModel (X_res [feat_to_keep], Y_res ,'Random Forest',RandomForestClassifier(), param, cv =10)


# In[45]:


param = {
            'C': [0.1,1,100,1000],
            'gamma':[0.0001,0.001, 0.005, 0.1,1, 3,5,10, 100]
         }
FitModel (X_res [feat_to_keep], Y_res,'SVC',SVC(), param, cv =5)


# In[46]:


param = { 'n_estimators': [100,500,1000,2000]  }
FitModel (X_res [feat_to_keep], Y_res,'XGBoost', XGBClassifier(),param, cv = 5)


# In[47]:


param = { 'n_estimators': [100,500,1000,2000]  }
FitModel (X_res, Y_res ,'Random Forest',RandomForestClassifier(), param, cv =10) 


# In[48]:


param = {
            'C': [0.1,1,100,1000],
            'gamma':[0.0001,0.001, 0.005, 0.1,1, 3,5,10, 100]
         }
FitModel (X_res, Y_res,'SVC',SVC(), param, cv =10)


# In[49]:


param = { 'n_estimators': [100,500,1000,2000]  }
FitModel (X_res, Y_res,'XGBoost', XGBClassifier(),param, cv = 10)


# In[50]:


load_model =pickle.load(open("XGBoost","rb"))


# In[51]:


pred1 = load_model.predict (x_test)
pred1


# In[52]:


load_model.best_params_


# In[53]:


print (accuracy_score (pred1,y_test))


# In[54]:


load_model =pickle.load(open("SVC","rb"))
pred1 = load_model.predict (x_test)
print (load_model.best_params_)
print (accuracy_score (pred1,y_test))
display (pred1)


# In[55]:


load_model =pickle.load(open("Random Forest","rb"))
pred1 = load_model.predict (x_test)
print (load_model.best_params_)
print (accuracy_score (pred1,y_test))
display (pred1)


# In[ ]:





# In[ ]:




