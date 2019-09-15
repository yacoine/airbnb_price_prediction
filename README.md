# Airbnb Price Prediction with Random Forest Regressor


## Personal mini-project
**Python3**
## Description

I found a comprehensive airbnb data set for listing in NYC in kaggle, source: (https://www.kaggle.com/dgomonov/new-york-city-airbnb-open-data) and I was interested to see if I could apply a random forest regressor for a price prediction.

After many manipulations of the features and the random forest regressor (RFR) attributes, I came up with a pretty solid prediction model based on 6 features and price range. The data-set includes qualitative and quantitative data, said qualitative features have been changed into byte-vectors with one-hot encoding method from pandas library ( get_dummies ). Only valid features have been considered and min_samples_split attrributes of the RFR has been altered because it converges to a better answer in a shorter amount of time.

## Next Step

Make a comprehensive GUI for the inputs. When user wants to find the predictive price of their listing, they will enter feature values of their proposed listing or already existing listing and will be given a price based on the trained model.

The data set uses 48,900 entries, and set model fitting will depend on the price range.



##GUI HELP NEEDED

The basic format of the gui should be 

**HOME PAGE**

Intro to project.

**First Page**

6 input fields.

minimum_night=input("Minimum nights")
number_of_reviews=input("Number of reviews")
reviews_per_month=input("Reviews per month")
calculated_host_listings_count=input("How many listings do you have, inclusive of this one.")
availability_365=input("How many days available per year")

neighbourhood_group=input("Neighbourhood")
Give 5 options for the neighbourhood in button format
o Manhattan, Brooklyn, Bronx, Queens, Staten Island
All values will be equal to 0, unless the button is selected than the value is =1

room_type=input("Room type (1=Entire home/apt, 2=Private room, 3=Share room ")
Give 3 options for the room type in button format
o Entire home/apt, Private room, Shared room
All values will be equal to 0, unless the button is selected than the value is =1

***I was also thinking that it would be interesting to not only restrict the price range to sets of
100, but also allow the user to adjust the price range, for example, instead of estimating that the price range of your listing should be 100 to 200, you can tinker with it and put 70-150. The predictive model has no problem with that change, it should just be included with the vars
upper_price_limit & lower_price_limit.***



##Difficulties

The difficulty arise when dealing with the price ranges in NYC because of extreme outsider data from luxury appartments. However, this is dealt with by specifiying price ranges of appartments and training a dummy model based on the price ranges and not the entire data.


***Please if you find any errors or whatnot let me know, I am open to all and any comments***

