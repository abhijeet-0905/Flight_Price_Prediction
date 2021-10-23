# Flight Price Prediction

[![git batch]](https://github.com/abhijeet-0905)
## Contents 
****
- [Overview](#overview)
- [Problem Statement](#Problem-Statement)
- [Dataset](#dataset)
- [Exploratory data analysis](#Exploratory-data-analysis)
- [Outlier detection and skewness treatment](#outlier)
- [Encoding the categorical data](#encode)
- [Scaling the data](#scaling)
- [Model Selection](#modelselect)
- [Cross-validation](#Cross-validation)
- [Model hyper-tuning](#Model-hyper-tuning)
- [Conclusion](#conclusion)
****
## Overview

## Problem Statement
Flight ticket prices can be something hard to guess, due to its ability of being reliant on various factors it keeps varying.
So in order to solve this problem, I have gathered a datset that contains prices of flight tickets for various airlines between the months of March and June of 2019 and between various cities, using which I aim to build a model which predicts the prices of the flights using various input features.

## Dataset
The [dataset][train_data] contains the features, along with the prices of the flights. It contains 10683 records, 10 input features and 1 output column — ‘Price’.

Here's a peek at the [dataset][train_data]:

<img src="https://github.com/abhijeet-0905/Flight_Price_Prediction/blob/main/zimages_dump/1_data_peek.JPG" alt="drawing" width="700">

###### Following is the description of features:

1.  **Airlines:** The name of the airline.
2.  **Date_of_Journey:** The date of the journey.
3.  **Source:** The source from which the service begins.
4.  **Destination:** The destination where the service ends.
5.  **Route:** The route taken by the flight to reach the destination.
6.  **Dep_Time:** The time when the journey starts from the source.
7.  **Arrival_Time:** Time of arrival at the destination.
8.  **Duration:** Total duration of the flight.
9.  **Total_Stops:** Total stops between the source and destination.
10. **Additional_Info:** Additional information about the flight.
11. **Price:** The price of the ticket.

## Exploratory data analysis
- Plotting countplots for categorical data :

<img src="https://github.com/abhijeet-0905/Flight_Price_Prediction/blob/main/zimages_dump/2_countplot_cat.png" alt="drawing" width="700">

  ##### **Insights:**
  1. **Airlines**:
      - Jet Airways is the most preferred airline with the highest count, followed by Indigo and AirIndia.
      - Count for Vistara Premium economy, Trujet, Multiple carries premium economy and Jet airways business is quite low.

  2. **Source:**
      - Majority of the flights take off from Delhi.
      - Chennai has the minimum count of flight take-offs.

  3. **Destination:**
      - Maximum flights land in Cochin.
      - Kolkata has the lowest count of receiving the flights.

  4. **Additional Info:**
      - Maximum rows have No info as the value.
      - I need to check how this column impacts the prices.

  5. **City1:**
      - City1 has same data as source column.
      - An additional value has been observed for ‘ DEL’, there is an extra Space in the name, count for this is very low. I will merge this with ‘DEL’.

  6. **City2:**
      - Majority of the flights take a stop in Bombay.
      - There are many cities with a very low count for stops. I will check how flights with 1stop impact prices of flights, and if any relation is there with stop place.

  7. **City3:**
      - Majority of the flights have no 2nd stop.
      - If there is a second stop, chances are high of the place being Cochin.
        
- Plotting distplots to check distribution in numeric data:

<img src="https://github.com/abhijeet-0905/Flight_Price_Prediction/blob/main/zimages_dump/3_distplot_cont.png" alt="drawing" width="700">

  ##### Insights:
  1. **Total stops:**
      - Majority of the flights have stops as 1, flights with 3 and 4 stops are quite low.

  2. **Date:**
      - There are no specific dates when the flights travel; the distribution is almost similar for all dates.

  3. **Month:**
      - People tend to travel less in April.
      - Flights in May and June have a higher count, seems like people travel during holiday months.

  4. **Year:**
      - This column has only 2019 as a value and can be dropped.

  5. **Dep_Time_Hour:**
      - Majority of the flights tend to fly in the early morning time.
      - Count of flights taking off during 16:00 - 23:00 is also high, Afternoon flights are less in number.

  6. **Dep_Time_Min:**
      - Most flights take off at whole hours (Mins as 00).

  7. **Arrival date:**
      - In majority of the cases, flights take off and land on the same day.

  8. **Arrival time hour:**
      - Majority of the flights reach its destination in the evening time around 18:00-19:00.
      - This seems to be because majority of the flights have take-off times in the morning and hence land after in the evening.

  9. **Arrival time min:**
      - This distribution is similar and does not give out any dedicated information.

  10. **Travel hours:**
      - Majority of the flights have travel time for around 2-3 hours, which seems ok since these are domestic flights.
      - Some flights have time around 30 hours too, this could be because of the number of stops in between.

  11. **Travel mins:**
      - The data is divided and is not pointing towards any specific points.

  12. **Price:**
      - The price column contains the minimum value as 1759 and maximum value as 79512.
      - Majority of the flights have price range between 1759–20k, and number of flights having prices greater than 20k are quite less.
      - Price range is skewed towards right.

- Plotting each categorical feature against Price:
<img src="https://github.com/abhijeet-0905/Flight_Price_Prediction/blob/main/zimages_dump/4_cat_col_vs_price.png" alt="drawing" width="700">

##### Insights:

- Jet airways business class has the highest prices between 50k — 80k.
- All the high cost flights depart from bangalore, rest of the flights have prices between 3k — 50k.
- All high cost flights have destination as Delhi, rest of the flights have prices between 3k — 50k.
- If a flight is of business class, its price would be high.
- The flights with high prices having 1 stop, have stop in Bombay.
- Flights with 2 stops, having higher prices, have stop in Delhi.

  - I seem to have quite less data where prices are higher than 40k.
  
  <img src="https://github.com/abhijeet-0905/Flight_Price_Prediction/blob/main/zimages_dump/_updated/09_price_40k.JPG" alt="drawing" width="700">
  
    - I have observed that the flights with high prices are 9 in number.
    - Majority of these flights fly from the same route — BLR->BOM->DEL.
    - Majority of the flights belong to Business class.
    - All the flights have Airlines as Jet airways.
    - All of these flights took flight in March.
        
- Plotting each numeric feature against Price:
<img src="https://github.com/abhijeet-0905/Flight_Price_Prediction/blob/main/zimages_dump/5_cont_col_vs_price.png" alt="drawing" width="700">

  ##### Insights:  
  - As number of stops increase, the price range gets decreasing into a smaller price window (10k — 22k).
  - High price flights are lesser during end of month.
  - Prices are higher in the month of March.
  - With increase in travel hours, price increases, but the number of flights decrease.

## Outlier detection and skewness treatment <a name="outlier"></a>
- Plotting boxplots to check the presence of outliers in our data:
<img src="https://github.com/abhijeet-0905/Flight_Price_Prediction/blob/main/zimages_dump/7_outliers.png" alt="drawing" width="700">

  ##### Insights:
  - Outliers are present in Total hours, Total stops and price.
  - I will not remove outliers from total stops since price is impacted by number of stops.
  - I will not remove the data with high number of hours, increase in number of hours shows a price pattern in the above graphs plotted for EDA.

- Skew:
<img src="https://github.com/abhijeet-0905/Flight_Price_Prediction/blob/main/zimages_dump/8_skew.JPG" alt="drawing" width="200">

  - I need to treat skewness for ‘Travel_hours’ column, considering a threshold value for skewness as +/-0.5 (we will not transform ‘Price’ column, since it is the target variable).
  - I have used log transformation method to remove skewness.

## Encoding the categorical data <a name="encode"></a>
- As Airlines contains Nominal data, One-Hot Encoding would be a good choice.
<img src="https://github.com/abhijeet-0905/Flight_Price_Prediction/blob/main/zimages_dump/9_encoding_nominal_airlines.JPG" alt="drawing" width="350">
- Other categorical columns consists of Ordinal data,hence LabelEncoding would do.
<img src="https://github.com/abhijeet-0905/Flight_Price_Prediction/blob/main/zimages_dump/9_encoding_ordinal_cols.JPG" alt="drawing" width="450">

## Scaling the data <a name="scaling"></a>

- Splitting the dataset into training set and test set prior to scaling is always a good choice as it prevents the data leakage.
-  Since there are certain columns with very small values and some columns with high values, scaling could bring the values of all the features within the same range.
-  **StandardScaler** follows Standard Normal Distribution (SND). Therefore, it makes mean = 0 and scales the data to unit variance.
<img src="https://github.com/abhijeet-0905/Flight_Price_Prediction/blob/main/zimages_dump/10_scaling.JPG" alt="drawing" width="300">

## Model Selection <a name="modelselect"></a>
- Since the target feature consists of continuous values, it is clearly a case of regression model. 
- I am going to fit the data into multiple regression models to compare the performance of all models and select the best model.

<img src="https://github.com/abhijeet-0905/Flight_Price_Prediction/blob/main/zimages_dump/11_fitting_the_model.JPG" alt="drawing" width="600">
<img src="https://github.com/abhijeet-0905/Flight_Price_Prediction/blob/main/zimages_dump/11_predicting.JPG" alt="drawing" width="650">

- I have obtained the first best score using **RandomForest Regressor**, with an **r2_score** of **78%**. I have also obtained the minimum values for mean_absolute_error, mean_squared_error and root_mean_squared_error (regression metrics) with this model.
<img src="https://github.com/abhijeet-0905/Flight_Price_Prediction/blob/main/zimages_dump/_updated/01_train_score_rf.JPG" alt="drawing" width="600">

- And then I have obtained the second best score using **GradientBoosting Regressor**, with an **r2_score** of **79%**.
<img src="https://github.com/abhijeet-0905/Flight_Price_Prediction/blob/main/zimages_dump/_updated/02_train_score_gb.JPG" alt="drawing" width="650">

- I will try to cross-validate both this models to check if the data is overfitting.

## Cross-validation
- I have perform the cross validation of my model to check if the model has any overfitting issue, by checking the ability of the model to make predictions on new data, using **k-folds**.
<img src="https://github.com/abhijeet-0905/Flight_Price_Prediction/blob/main/zimages_dump/13_cv.JPG" alt="drawing" width="350">
<img src="https://github.com/abhijeet-0905/Flight_Price_Prediction/blob/main/zimages_dump/_updated/03_cv_rf.JPG" alt="drawing" width="500">
<img src="https://github.com/abhijeet-0905/Flight_Price_Prediction/blob/main/zimages_dump/_updated/04_cv_gb.JPG" alt="drawing" width="500">

- The **Random Forest Regressor** provides a **cross validation score** of **79%**, and **gradient boosting regressor** also gives a **cross validation score** of **79%**. Let's hypertune both the models to check if the accuracy improves.

## Model hyper-tuning 
- I have applied GridSearch CV on both the models.

<img src="https://github.com/abhijeet-0905/Flight_Price_Prediction/blob/main/zimages_dump/14_gs.JPG" alt="drawing" width="300">

- It is a technique used to validate the model with different parameter combinations, by creating a grid of parameters and trying all the combinations to compare which combination gives the best results.


- RandomForest Regressor:
<img src="https://github.com/abhijeet-0905/Flight_Price_Prediction/blob/main/zimages_dump/14_gs_rf.JPG" alt="drawing" width="720">
<img src="https://github.com/abhijeet-0905/Flight_Price_Prediction/blob/main/zimages_dump/14_gs_rf_best_params.JPG" alt="drawing" width="450">
<img src="https://github.com/abhijeet-0905/Flight_Price_Prediction/blob/main/zimages_dump/_updated/05_hyp_tuning_rf_score.JPG" alt="drawing" width="140">

- GradientBoosting Regressor:
<img src="https://github.com/abhijeet-0905/Flight_Price_Prediction/blob/main/zimages_dump/14_gs_gb.JPG" alt="drawing" width="700">
<img src="https://github.com/abhijeet-0905/Flight_Price_Prediction/blob/main/zimages_dump/14_gs_gb_best_params.JPG" alt="drawing" width="170">
<img src="https://github.com/abhijeet-0905/Flight_Price_Prediction/blob/main/zimages_dump/_updated/06_hyp_tuning_gb_score.JPG" alt="drawing" width="140">

- The **r2_score** received for **Gradient Boosting Regressor** comes out to be better after hypertuning, which is **~84%**, as compared to **Random Forest Regressor** giving accuracy as **~83%**. The value of MAE also decreases, signifying that I was able to tune the model.
- Hence **Gradient Boosting Regressor** is my final model.

- I have saved [my model][saved_model] using joblib.

## Conclusion 
I now proceed to test the [object][saved_model] that I have saved using joblib.

<img src="https://github.com/abhijeet-0905/Flight_Price_Prediction/blob/main/zimages_dump/_updated/07_test_code.JPG" alt="drawing" width="450">

I have achieved an **r2_score** value of **85%**, meaning that the model is actually able to predict values quite near to the actual prices, for majority of the rows.

I also have a [test data][test_file] for which I need to predict the outputs.
I have loaded the test file, applied all the data modeling processes and operations on the [test data][test_file] similar to what I did with the [train data][train_data], and then make the final prediction using the [saved model object][saved_model].

<img src="https://github.com/abhijeet-0905/Flight_Price_Prediction/blob/main/zimages_dump/_updated/08_unseen_prediction.JPG" alt="drawing" width="400">

I was successfully able to train a regression model **Gradient Boosting Regressor** to predict the flights of prices with an **r2_score** of **85%**.

[![git batch]](https://github.com/abhijeet-0905)

[git batch]: <https://img.shields.io/badge/abhijeet--0905-FollowMe-blue>
[test_file]: <https://github.com/abhijeet-0905/Flight_Price_Prediction/blob/main/Test_set.xlsx>
[train_data]: <https://github.com/abhijeet-0905/Flight_Price_Prediction/blob/main/Data_Train.xlsx>
[saved_model]: <https://github.com/abhijeet-0905/Flight_Price_Prediction/blob/main/fare_prices_pred.obj>
