#airbnb_predict.py

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
import time
import matplotlib as mpl
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns # array of column names to encode

    def fit(self,X,y=None):
        return self # not relevant here

    def transform(self,X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):

	model= RandomForestRegressor(n_estimators=10,max_leaf_nodes=max_leaf_nodes, random_state=1)
	model.fit(train_X,train_y)
	predict_val=model.predict(val_X)
	mae=mean_absolute_error(predict_val,val_y)

	return mae
#Return mean absolute error with changes to minimum sampes split parameter
def get_mae_2(min_split, train_X, val_X, train_y, val_y):
 

	model= RandomForestRegressor(n_estimators=10,min_samples_split=min_split,random_state=1)
	model.fit(train_X,train_y)
	predict_val=model.predict(val_X)
	mae=mean_absolute_error(predict_val,val_y)

	return mae

house_data=pd.read_csv('AB_NYC_2019.csv').fillna(0)

long_lat=house_data.columns.tolist()

desired_features=['minimum_nights', 'number_of_reviews','reviews_per_month', 'calculated_host_listings_count', 'availability_365', 'neighbourhood_group_Bronx', 'neighbourhood_group_Brooklyn',
       'neighbourhood_group_Manhattan', 'neighbourhood_group_Queens',
       'neighbourhood_group_Staten Island', 'room_type_Entire home/apt',
       'room_type_Private room', 'room_type_Shared room']

house_data['neighbourhood_group']=pd.Categorical(house_data['neighbourhood_group'])
#house_data_dummy= pd.get_dummies(house_data['neighbourhood_group','room_type'], prefix='neigh_group')
house_data_dummy= pd.get_dummies(data=house_data, columns=['neighbourhood_group','room_type'])#, prefix='neigh_group')

house_data = pd.concat([house_data, house_data_dummy], axis=1)

print(house_data['room_type_Shared room'])
#onehotencoder = OneHotEncoder(categorical_features = 'neighbourhood_group')
#x = onehotencoder.fit_transform(house_data)

#print(house_data.columns)

#new_house_data=MultiColumnLabelEncoder(columns=['neighbourhood_group']).fit_transform(house_data)

#new_house_data=OneHotEncoder()
#print(a.neighbourhood_group)

#house_data.apply(LabelEncoder().fit_transform)

#values assigned for room type
#neighborhoods
"""le = LabelEncoder()
le.fit(house_data[desired_features])
print(house_data.head(10))"""
#print(le)
#print(list(le.classes_))
#house_data['neighborhood'] = labelencoder.fit_transform(house_data[:, 4])


y_train=house_data.price
X_train=house_data[desired_features]

train_X, val_X, train_y, val_y = train_test_split(X_train, y_train, random_state=1)

min=999_999_999
for i in range(2,200):
	mae=get_mae_2(i,train_X, val_X, train_y, val_y)

	if mae<min:
		min=mae
		index=i
		
	best_mae=min
	best_min_split=index

#best_min_split=70
#best_mae=get_mae_2(70,train_X, val_X, train_y, val_y)

print("Validation MAE for RFR with  {:,.0f} min sample split: {:,.0f}".format(best_min_split,best_mae))
#print(get_mae_2(4,train_X, val_X, train_y, val_y))

#print(desired_features)

#print(house_data.columns)