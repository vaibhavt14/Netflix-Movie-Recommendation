# Netflix-Movie-Recommendation: Project Overview 
* The given problem is a Recommendation problem 
* Predict the rating that a user would give to a movie that he has not yet rated
* For a given movie and user we need to predict the rating would be given by him/her to the movie. 
* Applied Surprise model,SVD(Singular value decomposition),SVDpp,xgboost regressor,item-item,user-user similarity,Matrix Factorization
* Performance metrics: Minimize the difference between predicted and actual rating (RMSE and MAPE)


## Code and Resources Used 
**Python Version:** 3.7  
**Packages:** pandas, numpy, sklearn, matplotlib, seaborn,nltk,scipy

## Dataset
Get the data from : https://www.kaggle.com/netflix-inc/netflix-prize-data/data
Data files :
* combined_data_1.txt
* combined_data_2.txt
* combined_data_3.txt
* combined_data_4.txt
* movie_titles.csv
-------------------------------------------------------------------------------
*	movie:Unique Id of movie
*	user:Unique id for each user
*	rating:Ratings given by user
*	date:User given rating during given date
* Dataset contain 480189 rows

## Data Cleaning
After understanding business requirements, I needed to clean it up so that it was usable for our model. I made the following changes and created the following variables:

* Merged all combined files and put in one csv file
*	Cleaned the duplicates rows 
*	Checking for null values 
* Implement some basics statatics such as mean,unique values
* Implement feature selection
* Splitting the dataset into train and test 
  

## EDA
I looked at the distributions of the data and the value counts for the various categorical variables and done some more data analysis such as.
- Distribution of ratings
- Number of ratings per month
- Analysis on the rating given by user
- Analysis of rating of a movie given by user
- Creating sparse matrix from dataframe
- Finding Global average of all movie ratings, Average rating per user, and Average rating per movie
- Finding average rating per user
- Finding average rating per movie.
- Cold start problem

![alt text](https://github.com/vaibhavt14/Netflix-Movie-Recommendation/blob/main/distribution%20of%20ratings.png "Class distribution of target variable")
![alt text](https://github.com/vaibhavt14/Netflix-Movie-Recommendation/blob/main/rating_per_month.png "Rating per month")
![alt text](https://github.com/vaibhavt14/Netflix-Movie-Recommendation/blob/main/rating_per_movie.png "Rating per movie")

## Model Building 

I split the random data into train and tests sets with a test size of 30% and took sample data as dataset has around large millions rows and applied cosine similarity

Used different models from surprise library such as SVD,Matrix Factorization,Baseline model and evaluated them using RMSE and MAPE 


## Model performance
The SVD far outperformed the other approaches on the test set. 
*	**SVD** : RMSE = 1.07260
*	**SVD++**: RMSE = 1.7284
*	**XG Boost**: RMSE = 1.0730




