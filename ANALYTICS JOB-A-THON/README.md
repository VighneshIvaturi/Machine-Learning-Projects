## ANALYTICS VIDHYA JOB-A-THON (19-11-2021 TO 21-11-2021)

## PRIVATE LEADERBOARD RANK-19, SCORE-0.729

# 1. Problem statement analysis
The given problem statement has demographic data about an organization/company who is 
facing rise in attrition. Losing employees in an organization is almost similar loosing a 
potential customer in the business and recruiting new employee is time taking cost to the 
company. By performing some meaningful analysis can result in reducing the attrition. The 
given data set contains some important factors like salary, quarterly ratings, education level
etc. These factors depict the employee’s situation in company and can result in predicting if 
that employee can leave or stay in the company. We will apply some data preprocessing 
techniques like scaling, encoding, feature engineering and apply several supervised machine learning 
models. Whichever model gives the best accuracy as per the data we will use that for predicting our 
test data.

# 2. Understanding target variable
The very first step is that the target variable is not specifically mentioned in the train data.
So, a small logic can be applied after going through the given data directory. There is 
particular variable named ‘LastWorkingDate’, Which has 91.5% of null values. These null 
values are not actually the missing data they indicate that the particular company has 
discontinued the work and has left the company. If the variable has any date record that 
means the employee has left the organization on that particular date. So, we can create a 
binary class variable, by keeping 0 where last working date column is null and 1 where last 
working date column has a date. 

# 3. Creating the test dataset
The given test dataset contains only the one column with employee id. Thus, for taking the 
performance and predicting the attrition of employees, we will manipulate the dataframe by 
joining (“inner join”) function between test and train to get those parameters in the same 
order as it is arranged in test data. After that we will predict the status of employees as per 
the performance parameters. Finally, we shall put the predictions in the submission file and 
finally upload the solution.

#4. Data Preprocessing

I. After reading the dataset we can observe that there are multiple datapoints with 
same Employee id. This may be due to different business values and mainly due 
to the update of designation and salary of the particular employee So we can 
group the employee id except the total business value column and take the last Id 
of the employee and create a dataframe. Then we will create another dataframe by
grouping the ‘Total Business Value’ add them in each group of employee id and 
finally concatenate both the dataframes

II. We will perform EDA for all the variables by plotting several graphs. For 
categorical variables we will plot count plot, bar plot, pie chart and for the 
numerical variables we will plot histogram, scatterplot and box plot
III. If we observe the education level variable there are three categories in it. They are 
college, masters, bachelors. We can combine the masters and bachelors sub 
categories into a single sub category i.e., graduate

IV. The target variable in the dataset is imbalanced as there are a greater number of 
0’s than 1’s. SO we shall balance the data using SMOTE technique and perform 
modelling

V. We shall encode the categorical columns which are in text format as they provide 
more meaningful instance for model training and prediction.
VI. If we observe the age and salary column. The age data points are in two digit and 
the salary data points are in 6 and 7 digits. So, they should be brought down to a 
particular scale. Hence numerical variables should be scaled as the data between 
variables has more variations

# 5. Model Building and Predictions
I. After performing the above data preprocessing and feature engineering steps the 
final dataframe will be further split into test and train sets

II. Classification algorithms like Logistic Regression, Decision tree, Random Forest, 
Adaboost classifier, Xgboost classifier, Support vector classifier are used for 
training the model and tested

III. PCA is also applied and reduced the variables to 6 and applied to the model using 
support vector classifier

IV. All the models and their accuracy scores are displayed in a separate dataframe

V. Predictions of Random Forest, Xgboost, Support vector classifier are a total of 14 
files are submitted for calculating the scores. Random Forest has given 0.66 score 
and SVM has given 0.73 score which is the highest among all

VI. Support Vector Classifier model with C value of 1 and gamma value auto and 
linear kernel is selected and fitted with the data. SVM has several advantages
among other machine learning models. It has availability of soft margin. It
intentionally performs misclassification of some data records so as to go through
any kind of data
