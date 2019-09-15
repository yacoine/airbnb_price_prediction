#airbnb_predict.py

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline

import time
import matplotlib.pyplot as plt
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

#Label encoder for multiple lines
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

# Give the mean average error value of a random forest regressor by changing the
# max_leaf_nodes param
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

# TO DO INPUT VALUES for price prediction 
"""
minimum_night=input("Minimum nights")
number_of_reviews=input("Number of reviews")
reviews_per_month=input("Reviews per month")
calculated_host_listings_count=input("How many listings do you have, inclusive of this one.")
availability_365=input("How many days available per year")
neighbourhood_group=input("Neighbourhood")
room_type=input("Room type (1=Entire home/apt, 2=Private room, 3=Share room ")
"""



price_lower_limit=400
price_upper_limit=500

#this selects prices of houses in ranges between the upper and lower limits 
house_data=house_data.loc[(house_data['price'] >= price_lower_limit) & (house_data['price'] <= price_upper_limit)]


#features desired to be used for the random forest regressor
desired_features=['minimum_nights', 'number_of_reviews', 'calculated_host_listings_count', 'availability_365', 'neighbourhood_group_Bronx', 'neighbourhood_group_Brooklyn',
       'neighbourhood_group_Manhattan', 'neighbourhood_group_Queens',
       'neighbourhood_group_Staten Island', 'room_type_Entire home/apt',
       'room_type_Private room', 'room_type_Shared room']

#Pandas version of hot one encoder, splitting qualitative data into binary data
#This might not be the most effective way to create a predictive model
#but i believe it is the best for random forest tree regressor
#house_data['neighbourhood_group']=pd.Categorical(house_data['neighbourhood_group'])
house_data= pd.get_dummies(data=house_data, columns=['neighbourhood_group','room_type'])#, prefix='neigh_group')



y_train=house_data.price
X_train=house_data[desired_features].abs()

train_X, val_X, train_y, val_y = train_test_split(X_train, y_train, random_state=1)

#This is used to find the best parameter by looping through possible min_split_leaf
# or looping through possible max_leaf_nodes, depending on which get_mae/get_mea_2 function
#you decided to use. I believe that get_mea_2 (min split leaf) is more robust in its mae

"""
min=999_999_999
for i in range(2,100):
	mae=get_mae_2(i,train_X, val_X, train_y, val_y)

	if mae<min:
		min=mae
		index=i
		
	best_mae=min
	best_min_split=index


print(house_data.price.mean())
"""

#Uncomment the below two lines of code if you find a param that is better and comment the for loop
best_min_split=75
best_mae=get_mae_2(best_min_split,train_X, val_X, train_y, val_y)

print("Validation MAE for RFR with  {:,.0f} min sample split: +/- ${:,.0f}".format(best_min_split,best_mae))

minimum_nights=2
number_of_reviews=10
calculated_host_listings_count=1
availability_365=250
ng_Bronx=0
ng_Manhattan=0
ng_Brooklyn=0
ng_Queens=1
ng_Staten=0
rt_private=0
rt_shared=0
rt_entire_home=1

prediction_features={'minimum_nights':[minimum_nights], 'number_of_reviews':[number_of_reviews], 'calculated_host_listings_count':[calculated_host_listings_count], 
	   'availability_365':[availability_365], 'neighbourhood_group_Bronx':[ng_Bronx], 'neighbourhood_group_Brooklyn':[ng_Brooklyn],
       'neighbourhood_group_Manhattan':[ng_Manhattan], 'neighbourhood_group_Queens':[ng_Queens],
       'neighbourhood_group_Staten Island':[ng_Staten], 'room_type_Entire home/apt':[rt_entire_home],
       'room_type_Private room':[rt_private], 'room_type_Shared room':[rt_shared]}

prediction_df=pd.DataFrame(prediction_features)


model1= RandomForestRegressor(n_estimators=10,min_samples_split=75, random_state=1)
model1.fit(X_train,y_train)
predict_price=model1.predict(prediction_df)



print("predictive price")
print(predict_price)
"""
plt1.style.use('ggplot')
plt1.ylabel("Number of houses")
plt1.xlabel("Neighbourhoods NYC")
plt1.title("Airbnb Offers per Neighbourhood NYC")
plt1.hist(house_data['neighbourhood_group'], bins=10)
"""
"""
plt.figure(1)
plt.style.use('ggplot')
plt.ylabel("Number of houses")
plt.xlabel("Neighbourhoods NYC")
plt.title("Airbnb Offers per Neighbourhood NYC")
plt.hist(house_data['neighbourhood_group'], bins=10)
"""
"""plt.figure(2)
plt.style.use('ggplot')
plt.ylabel("Price per night")
plt.xlabel("Nnumber of reviews")
plt.title("Airbnb Offers per Neighbourhood NYC")
plt.scatter(house_data['number_of_reviews'],bins=100)"""

#plt.subplot(1,2,1)

#fig.subplots_adjust(hspace=0.4, wspace=0.4)
#fig.style.use('ggplot')

"""model1= RandomForestRegressor(n_estimators=10,min_samples_split=96, random_state=1)
model1.fit(train_X,train_y)
predict_price=model1.predict(X_train)

plt.figure(3)
plt.style.use('ggplot')
plt.ylabel("Price per night")
plt.xlabel("ID")
plt.title("price comp.")
plt.scatter(house_data['id'],abs(house_data.price-predict_price))

"""









"""
plt1.ylabel("Number of houses")
plt1.xlabel("Neighbourhoods NYC")
plt1.title("Airbnb Offers per Neighbourhood NYC")
plt1.hist(house_data['neighbourhood_group'], bins=10)
"""

plt.show()


