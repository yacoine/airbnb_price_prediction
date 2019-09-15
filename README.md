# Airbnb Price Prediction with Random Forest Regressor

## Personal mini-project

## Description

I found a comprehensive airbnb data set for listing in NYC in kaggle, source: (https://www.kaggle.com/dgomonov/new-york-city-airbnb-open-data) and I was interested to see if I could apply a random forest regressor for a price prediction.

After many manipulations of the features and the random forest regressor (RFR) attributes, I came up with a pretty solid prediction model based on 6 features and price range. The data-set includes qualitative and quantitative data, said qualitative features have been changed into byte-vectors with one-hot encoding method from pandas library ( get_dummies ). Only valid features have been considered and min_samples_split attrributes of the RFR has been altered because it converges to a better answer in a shorter amount of time.

## Next Step

Make a comprehensive GUI for the inputs. When user wants to find the predictive price of their listing, they will enter feature values of their proposed listing or already existing listing and will be given a price based on the trained model.

The data set uses 48,900 entries, and set model fitting will depend on the price range.

##Difficulties

The difficulty arise when dealing with the price ranges in NYC because of extreme outsider data from luxury appartments. However, this is dealt with by specifiying price ranges of appartments and training a dummy model based on the price ranges and not the entire data.


