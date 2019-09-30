#airbnb_predict.py

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
import seaborn as sns
from numpy import *
import time #not used by helpful for RF regressor attribute choices during looping
import matplotlib.pyplot as plt
from warnings import simplefilter
import sys
from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5 import uic, QtCore

simplefilter(action='ignore', category=FutureWarning) #sklearn warning feature for updates is ignored for looping


#https://github.com/eyllanesc/stackoverflow/tree/master/questions/44603119
#This is used in case I wanted to ever print a data fram through PyQt5
#Credit is given to the above git path
class DataFrameModel(QtCore.QAbstractTableModel):
    DtypeRole = QtCore.Qt.UserRole + 1000
    ValueRole = QtCore.Qt.UserRole + 1001

    def __init__(self, df=pd.DataFrame(), parent=None):
        super(DataFrameModel, self).__init__(parent)
        self._dataframe = df

    def setDataFrame(self, dataframe):
        self.beginResetModel()
        self._dataframe = dataframe.copy()
        self.endResetModel()

    def dataFrame(self):
        return self._dataframe

    dataFrame = QtCore.pyqtProperty(pd.DataFrame, fget=dataFrame, fset=setDataFrame)

    @QtCore.pyqtSlot(int, QtCore.Qt.Orientation, result=str)
    def headerData(self, section: int, orientation: QtCore.Qt.Orientation, role: int = QtCore.Qt.DisplayRole):
        if role == QtCore.Qt.DisplayRole:
            if orientation == QtCore.Qt.Horizontal:
                return self._dataframe.columns[section]
            else:
                return str(self._dataframe.index[section])
        return QtCore.QVariant()

    def rowCount(self, parent=QtCore.QModelIndex()):
        if parent.isValid():
            return 0
        return len(self._dataframe.index)

    def columnCount(self, parent=QtCore.QModelIndex()):
        if parent.isValid():
            return 0
        return self._dataframe.columns.size

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if not index.isValid() or not (0 <= index.row() < self.rowCount() \
            and 0 <= index.column() < self.columnCount()):
            return QtCore.QVariant()
        row = self._dataframe.index[index.row()]
        col = self._dataframe.columns[index.column()]
        dt = self._dataframe[col].dtype

        val = self._dataframe.iloc[row][col]
        if role == QtCore.Qt.DisplayRole:
            return str(val)
        elif role == DataFrameModel.ValueRole:
            return val
        if role == DataFrameModel.DtypeRole:
            return dt
        return QtCore.QVariant()

    def roleNames(self):
        roles = {
            QtCore.Qt.DisplayRole: b'display',
            DataFrameModel.DtypeRole: b'dtype',
            DataFrameModel.ValueRole: b'value'
        }
        return roles


#Label encoder for multiple lines
ui_name= 'trial.ui'
Ui_MainWindow, QtBaseClass = uic.loadUiType(ui_name)

class MyApp(QMainWindow):
	minimum_night=''
	number_reviews=''
	current_number_listings=''
	availability_365=''
	neigh=''
	room_type=''
	lower_limit=''
	upper_limit=''

	def __init__(self):
		super(MyApp, self).__init__()
		self.ui = Ui_MainWindow()
		self.ui.setupUi(self)
		self.ui.pushButton.clicked.connect(self.values_stored)

		
		#print(self.ui.minimum_nights)
        

	def values_stored(self):

		ng_Brooklyn=ng_Bronx=ng_Manhattan=ng_Staten=ng_Queens =0
		room_type_Entire=room_type_Private=room_type_Shared=0

		
		minimum_night=float(self.ui.minimum_nights.text())
		number_reviews=int(self.ui.number_reviews.text())
		current_number_listings=int(self.ui.current_number_listings.text())
		availability_365=int(self.ui.availability_365.text())
		neigh=int(self.ui.neighbourhood_val.currentIndex())
		if(neigh==0):
			ng_Brooklyn=1
		elif(neigh==1):
			ng_Bronx=1
		elif(neigh==2):
			ng_Manhattan=1
		elif(neigh==3):
			ng_Staten=1
		else:
			ng_Queens=1

		room_type=int(self.ui.room_type.currentIndex())
		if(room_type==0):
			room_type_Entire=1
		elif(room_type==1):
			room_type_Private=1
		else:
			room_type_Shared=1

		lower_limit=int(self.ui.lower_limit.value())
		upper_limit=int(self.ui.upper_limit.value())

		prediction_features={'minimum_nights':[minimum_night], 'number_of_reviews':[number_reviews], 'calculated_host_listings_count':[current_number_listings], 
	   'availability_365':[availability_365], 'neighbourhood_group_Bronx':[ng_Bronx], 'neighbourhood_group_Brooklyn':[ng_Brooklyn],
       'neighbourhood_group_Manhattan':[ng_Manhattan], 'neighbourhood_group_Queens':[ng_Queens],
       'neighbourhood_group_Staten Island':[ng_Staten], 'room_type_Entire home/apt':[room_type_Entire],
       'room_type_Private room':[room_type_Private], 'room_type_Shared room':[room_type_Shared]}

		self.ui.price_return.setText(predict_price(prediction_features,lower_limit,upper_limit))

		#predict_price(prediction_features)

		return prediction_features

	def update(self, app):

		app.price_lower_limit=int(self.ui.lower_limit.value())
		app.price_upper_limit=int(self.ui.upper_limit.value())





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

# Return the mean average error value of a random forest regressor by changing the
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


def predict_price(features,lower,upper):

		price_lower_limit=lower
		price_upper_limit=upper


		prediction_df=pd.DataFrame(features)
		house_data=pd.read_csv('AB_NYC_2019.csv').fillna(0)
		house_data=house_data.loc[(house_data['price'] >= price_lower_limit) & (house_data['price'] <= price_upper_limit)]
		print(house_data.price)

		desired_features=['minimum_nights', 'number_of_reviews', 'calculated_host_listings_count', 'availability_365', 
	   'neighbourhood_group_Bronx', 'neighbourhood_group_Brooklyn',
       'neighbourhood_group_Manhattan', 'neighbourhood_group_Queens',
       'neighbourhood_group_Staten Island', 'room_type_Entire home/apt',
       'room_type_Private room', 'room_type_Shared room']

		
		house_data= pd.get_dummies(data=house_data, columns=['neighbourhood_group','room_type'])
		X_train=house_data[desired_features].abs()
		y_train=house_data.price

#fitting and prediction of the model
		model1= RandomForestRegressor(n_estimators=10,min_samples_split=75, random_state=1)
		model1.fit(X_train,y_train)
		predict_price=model1.predict(prediction_df)
		#MyApp.price_return.setText("predict_price")

		#MyApp.price_return.setText(" WORKS ?")
		return str(predict_price[0])






house_data=pd.read_csv('AB_NYC_2019.csv').fillna(0)
graph_house_data=house_data.copy()

#This can be uncommented or  changed if you wanted to work with the model itself
price_lower_limit=100
price_upper_limit=200


#this selects prices of houses in ranges between the upper and lower limits 
house_data=house_data.loc[(house_data['price'] >= price_lower_limit) & (house_data['price'] <= price_upper_limit)]


#features desired to be used for the random forest regressor
desired_features=['minimum_nights', 'number_of_reviews', 'calculated_host_listings_count', 'availability_365', 
	   'neighbourhood_group_Bronx', 'neighbourhood_group_Brooklyn',
       'neighbourhood_group_Manhattan', 'neighbourhood_group_Queens',
       'neighbourhood_group_Staten Island', 'room_type_Entire home/apt',
       'room_type_Private room', 'room_type_Shared room']

#Pandas version of hot one encoder, splitting qualitative data into binary data
#This might not be the most effective way to create a predictive model
#but i believe it is the best for random forest tree regressor
#house_data['neighbourhood_group']=pd.Categorical(house_data['neighbourhood_group'])



house_data= pd.get_dummies(data=house_data, columns=['neighbourhood_group','room_type'])#, prefix='neigh_group')


#Price of all listings
y_train=house_data.price


print(DataFrameModel(y_train))

#All attributes of the model with the desired features
X_train=house_data[desired_features].abs()


#Splitting data into training set and testing set
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

#Uncomment the below two lines of code if you find a param that is better and comment the above for loop
best_min_split=75
best_mae=get_mae_2(best_min_split,train_X, val_X, train_y, val_y)




print("Validation MAE for RFR with  {:,.0f} min sample split: +/- ${:,.0f}".format(best_min_split,best_mae))
"""

#creationg of a df for the inputted values
prediction_df=pd.DataFrame(prediction_features)

#fitting and prediction of the model
model1= RandomForestRegressor(n_estimators=10,min_samples_split=75, random_state=1)
model1.fit(X_train,y_train)
predict_price=model1.predict(prediction_df)

"""

#Uncomment the triple quotations in order to not dispay the graphs
#"""
f,ax = plt.subplots(figsize=(16,8))
ax = sns.scatterplot(y=graph_house_data.latitude,x=graph_house_data.longitude,hue=graph_house_data.neighbourhood_group,palette="coolwarm")
plt.show()

f,ax = plt.subplots(figsize=(16,8))
ax = sns.scatterplot(y=graph_house_data.latitude,x=graph_house_data.longitude,hue=graph_house_data.price,palette="coolwarm", hue_norm=(0, 300))
plt.show()
#"""
app = QApplication(sys.argv)
window = MyApp()
window.show()

print('upper limit:')
print(price_lower_limit)
sys.exit(app.exec_())

