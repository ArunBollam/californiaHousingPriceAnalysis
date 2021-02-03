#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os
import tarfile
from six.moves import urllib

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = "datasets/housing"
HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH + "/housing.tgz"


# In[7]:


def fetch_housing_data(housing_url = HOUSING_URL, housing_path = HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


# In[10]:


fetch_housing_data()


# In[11]:


import pandas as pd

def load_housing_data(housing_path = HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


# In[12]:


housing = load_housing_data()
housing.head()


# In[13]:


housing.info()


# In[14]:


housing['ocean_proximity'].value_counts()


# In[15]:


housing.describe()


# In[16]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
housing.hist(bins = 50, figsize = (20,15))
plt.show()


# In[19]:


import numpy as np

def split_train_test(data, test_ratio):
    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


# In[20]:


train_set, test_set = split_train_test(housing, 0.2)
print(len(train_set), "train +", len(test_set), "test") 


# In[22]:


#random train test split
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size = 0.2, random_state = 42)


# In[23]:


#stratified sampling based on the median_income
housing["income_cat"] = np.ceil(housing["median_income"]/1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace= True)


# In[ ]:


from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
 strat_train_set = housing.loc[train_index]
 strat_test_set = housing.loc[test_index]


# In[25]:


from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]


# In[32]:


housing["income_cat"].value_counts()/len(housing)


# In[33]:


for set in (strat_train_set, strat_test_set):
    set.drop(["income_cat"], axis =1, inplace = True)


# In[34]:


#Visualization
housing = strat_train_set.copy()


# In[36]:


housing.plot(kind = "scatter", x = "longitude", y = "latitude", figsize = (10, 8))


# In[37]:


housing.plot(kind = "scatter", x = "longitude", y = "latitude", alpha = 0.1, figsize = (10,8))


# In[ ]:


housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
 s=housing["population"]/100, label="population",
 c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
)
plt.legend()


# In[40]:


housing.plot(kind = "scatter", x = "longitude", y = "latitude", alpha = 0.4, 
            s = housing["population"]/100, label = "population", 
            c = "median_house_value", cmap=plt.get_cmap("jet"), colorbar = True, figsize = (10,8))
plt.legend()


# In[43]:


#Correlations
corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending = False)


# In[ ]:


from pandas.tools.plotting import scatter_matrix
attributes = ["median_house_value", "median_income", "total_rooms",
 "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))


# In[48]:


from pandas.plotting import scatter_matrix
attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
scatter_matrix(housing[attributes], figsize= (12,8))


# In[51]:


housing.plot(kind = "scatter", x = "median_income", y = "median_house_value", alpha = 0.1)


# In[52]:


housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedroooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"] = housing["population"]/housing["households"]


# In[53]:


corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending = False)


# In[54]:


#Preparing the data for Machine Learning Algorithms
housing = strat_train_set.drop("median_house_value", axis = 1)
housing_labels = strat_train_set["median_house_value"].copy()


# In[57]:


#Data Cleaning
from sklearn.impute import SimpleImputer 
imputer = SimpleImputer(strategy='median')


# In[58]:


#dataset with only numerical attributes
housing_num = housing.drop("ocean_proximity", axis = 1)


# In[59]:


imputer.fit(housing_num)


# In[60]:


imputer.statistics_


# In[61]:


housing_num.median().values


# In[63]:


X = imputer.transform(housing_num)
housing_tr = pd.DataFrame(X, columns=housing_num.columns)


# In[66]:


#Handling Text and Categorical attributes
from sklearn.preprocessing import LabelBinarizer
encoder = LabelBinarizer()
housing_cat = housing["ocean_proximity"]
housing_cat_1hot = encoder.fit_transform(housing_cat)
housing_cat_1hot


# In[71]:


#Custom Transformers
from sklearn.base import BaseEstimator, TransformerMixin

rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    
    def __init__(self, add_bedrooms_per_room = True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
        
    def fit(self, X, y = None):
        return self
        
    def transform(self, X, y = None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix]/X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix]/X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else: 
            return np.c_[X, rooms_per_household, population_per_household]
        
attr_adder = CombinedAttributesAdder(add_bedrooms_per_room = False)
housing_extra_attribs = attr_adder.transform(housing.values)


# In[73]:


housing_extra_attribs


# In[80]:


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
        
    def fit(self, X, y = None):
        return self
    
    def transform(self, X):
        return X[self.attribute_names].values


# In[83]:


class CustomLabelBinarizer(BaseEstimator, TransformerMixin):
    def __init__(self, sparse_output=False):
        self.sparse_output = sparse_output
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        enc = LabelBinarizer(sparse_output=self.sparse_output)
        return enc.fit_transform(X)


# In[75]:


#Transformation Pipelines --housing_num is training data
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy = "median")), 
    ('attribs_adder', CombinedAttributesAdder()), 
    ('std_scaler', StandardScaler()), 
])

housing_num_tr = num_pipeline.fit_transform(housing_num)


# In[84]:


#Full pipeline to transform both numerical and categorical attributes
from sklearn.pipeline import FeatureUnion

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

num_pipeline = Pipeline([
    ('selector', DataFrameSelector(num_attribs)), 
    ('imputer', SimpleImputer(strategy = "median")), 
    ('attribs_adder', CombinedAttributesAdder()), 
    ('std_scaler', StandardScaler()), 
])

cat_pipeline = Pipeline([
    ('selector', DataFrameSelector(cat_attribs)), 
    ('label_binarizer', CustomLabelBinarizer()), 
])

full_pipeline = FeatureUnion(transformer_list = [
    ("num_pipeline", num_pipeline), 
    ("cat_pipeline", cat_pipeline), 
])


# In[100]:


housing_prepared = full_pipeline.fit_transform(housing)
housing_prepared.shape


# In[90]:


#Training and evaluating on Training dataset
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)


# In[108]:


some_data = housing.iloc[:10000]
some_lables = housing_labels.iloc[:10000]
some_data_prepared = full_pipeline.transform(some_data)
some_data_prepared.shape


# In[105]:


some_data_prepared


# In[110]:


from sklearn.metrics import mean_squared_error
housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse


# In[111]:


#Decision Tree
from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)


# In[112]:


housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse

