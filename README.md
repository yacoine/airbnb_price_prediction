# Airbnb Price Prediction with Random Forest Regressor


## Personal mini-project
**Python3**
**Taught myself the RFR model, most of the features, and the GUI in around 3ish days**
## Description

I found a comprehensive airbnb data set for listings in NYC which lead my curious side see if I could apply a random forest regressor for a price prediction.

After manipulating the features and the random forest regressor (RFR) attributes multiple times, I came up with a pretty solid prediction model based on 6 features and price range. The data-set includes qualitative and quantitative data, said qualitative features have been changed into byte-vectors with one-hot encoding method from pandas library ( get_dummies ). Only valid features have been considered for this predictive model, as well as, a change in the RGR's attribute- min_samples_split. The reason for picking this attribute over others is further explained in this repo.

This price predictive model can be useful for individuals that own airbnb listings and or for individuals interested in seeing what their listing would be worth. This predictive model in my opinion could be greatly improved if the amount of rooms, number of guests, number of bathrooms, and other quantitative data was offered. However, even with such a restricted number of features, the model still manages to be good, proving the efficiency of RFRs.

## Next Step

Finding a correlating data set that includes space measurements of the listings with more quantitative description regarding
the listings. However, this shows the strength of using RFRs on data sets that do not seem very promising.

The data set uses 48,900 entries and  model fitting depends on price range.



## GUI HELP NEEDED

GUI works! No help needed.



## Difficulties

The difficulty arise when dealing with the price ranges in NYC because of extremely flamboyantly priced  luxury appartments. However, this is dealt with by specifiying price ranges of appartments and training a dummy model based on the price ranges and not the entire data.

## Code Explained

Data set used comes from https://www.kaggle.com/dgomonov/new-york-city-airbnb-open-data.

Data set is divided by these columns:

![Column features from .csv file](images/column_names.png)

The set-up of the data frame looks like this:

![Data frame setup from .csv file](images/df_setup.png)

The .csv file for the airbnb data has both qualitative and quantitative data, which is not the most ideal when working with random forest regressors, because the qualitative data in itself does not hold any mathematical meaning. For example, if we were to give Manhattan=1 and Bronx=2, the model will make false predictions based on the value that these two values hold. Which is not a sound when dealing with qualitative data. Hence, it was imperative to change, pick, and choose which of the features would be most ideal, in order to train our model to predict a listing's price. The most coherent features were taken into consideration. For example; number of total reviews is more important in this case than the number of reviews per month, as it does not add to the model's accuracy and only slows down the model.

In order to take into consideration the qualitative data it was converted to a binary vector using pandas' version of one hot encoder, get_dummies. This allows data such as type of neighbourhood: ["Bronx" , "Brooklyn", "Manhattan", ...] to be stored as 0's and 1's. Zero if the neighbourhood does not belong and one if it does. This was applied to two qualitative features; the neighbourhood group and the room type: ["Entire apt/home", "Share room", "Private room"].

Method: 
>house_data= pd.get_dummies(data=house_data, columns=['neighbourhood_group','room_type'])

The desired features were then split in order to create a training data set for the model and a test data set.

![Split data](images/split_data.png)


Two functions were created in order to find the best attribute to change for the RFR.
Forest tree regressor's attribute chosen were max_leaf_nodes, which allocates a maximum amount of leaves per node and min_samples_split, which is the minimum number of samples required to split an internal node. These two functions have a parameter max_leaf_nodes or min_split that allows looping in the main() in order to find the best value, and returns the mean absolute error for that model.

![functions](images/function.png)


Little graphs:

This graph shows the position of listings with color coded price ranges for values up to $300.
![price per location](images/price_loc.png)



Another fancy graph, that is very much self-explanatory. 
![neighbourhoods](images/neigh_loc.png)



In order to test if the model works, I compared attributes from an actual listing online to what my model would predict.

This is the posting I used to find the price range.

![website](images/website_test.png)

Which resulted in these attributes.

![attributes](images/test_numbers.png)

We can also see that as the price range becomes more specific the model converges towards a better answer.

With the above attributes and a model fitted and trained by the .csv file, with min_samples_split=75, n_estimator=10 the model predicted these values;

|               Price range               |         Predicted price          |
| --------------------------------------- | -------------------------------- |
|                  0:400                  |              $225                | 
|                  0:300                  |              $172.52             |
|                  0:200                  |              $145.23             | 
|                  100:200                |              $150.75             | 



With a price range of $100:$200 the prediction of the $150 listing was **$150.75 !!!!**

We can clearly see that as the price range becomes more specific the price prediction becomes more accurate. In this case our prediction was $0.75 off, but the mean average error was +/- $25. As the upper and lower limit of price is made more specific the training model is mostly trained with listing in that price range and is able to determine more accurately. 

## Setup
Special thanks to https://github.com/ademilly for his help on the set-up tools.


Using the virtual environment package `venv`:
```bash
    $ python3 -m venv venv
    $ source venv/bin/activate
    $ pip install -r requirements.txt
```

## run

```bash
    $ python3 airbnb_predict.py
```

***Please if you find any errors or whatnot let me know, I am open to all and any comments***

email: y.acoine@gmail.com
