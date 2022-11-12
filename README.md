# Feature Engineering

<h4>Learn the steps and principles of creating better features for a ML model<h4></br>
<h4> A process of extracting useful features from raw data using Maths, Statisics, Domain knowledge.</h4>

Feature engineering helps in data cleaning,
We do feature engineering before the data is put in machine learning model


[Feature Engineering Kaggle](https://www.kaggle.com/code/ryanholbrook/what-is-feature-engineering)<br/>
[EDA, Step by step](https://www.simplilearn.com/tutorials/data-analytics-tutorial/exploratory-data-analysis)<br/>
[Data Cleaning](https://towardsdatascience.com/what-is-feature-engineering-importance-tools-and-techniques-for-machine-learning-2080b0269f10)<br/>
[Kaggle data cleaning course and codes](https://www.kaggle.com/learn/data-cleaning)<br />
[IBM_DataCleaning_Lab](https://www.coursera.org/learn/ibm-exploratory-data-analysis-for-machine-learning/ungradedLti/qfAqI/practice-lab-data-cleaning)

 
 ![dataCleaning2](https://user-images.githubusercontent.com/33677647/201374239-ca5f3c23-ae74-4275-b413-6bf760090caf.jpg)


You might perform feature engineering to:
- improve a model's predictive performance
- reduce computational or data needs
- improve interpretability of the results

some of its examples are

- Check which are most important features.
- Outlier detection
- Imputation = the process of replacing missing data with substituted values.  Handling missing values
- Feature encoding (One hot encoding/ Target encoding) 
- New Feature creation = Two features as one using domain knowledge, it saves time/ memory, Price and sq.ft --> price/sq.ft
- Feature transformation
- Feature Scaling
- Segmentation features creation with k-means clustering.
- Principal component analysis = decompose a dataset's variation into features with PCA
- Check (If we intend to use linear regression as our model, then we need to check if there is linear relation btw. our features or not, is data normalized).


![a](https://user-images.githubusercontent.com/33677647/201405617-ec201038-2140-4521-b289-e39ef17b9207.JPG)

## 1_Check Linear Relation if intended 

For a feature(column) to be useful, it must have a relationship to the target variable. So that our model can learn something from that variable in predicting the target variable, When using a linear model, we have to keep in mind that our goal is to transform the features (transform the features in a way) to make their relationship to the target linear. 

 we can actually transform features to ensure a linear relationship. 
**i.e log transformations can be a useful way to find a linear relationship when the underlying raw data may not actually have a linear relationship.**
 
![a](https://user-images.githubusercontent.com/33677647/201498317-c70596f7-9725-4266-bc4a-0ce3b48c36bf.JPG)

So we create a new feature in dataset by squaring the feature lenght i.e Area = L*L, So area feature is a higher degree polynomial of lenght feature, which is now fitting the plot well.

 ![b](https://user-images.githubusercontent.com/33677647/201498318-d6c7363f-6593-4e3f-bbbe-9b5b707b9d92.JPG)

<i><b>Tip</b></i>---> why data is needed to be normally distributed ?
Ans : We convert normal distributions into the standard normal distribution for several reasons: To find the probability of observations in a distribution falling above or below a given value. To find the probability that a sample mean significantly differs from a known population mean.(As far as I understand it is that we compare the mean of population to mean of sample to see the accuracy of our sample)

### Feature Transformation (Transformation of distribution of data / to make linear relationship between features)

For linear regression, we assue that our residuals are normally distributed. But often raw data and target predicted feature is skewed,
log transformations can be a useful way to find a linear relationship when the underlying raw data may not actually have a linear relationship.
 
![a](https://user-images.githubusercontent.com/33677647/201408901-9f0fec30-949b-4479-8d5b-57ee6216564e.JPG)

some libraries to transfom data :
- **log and log1p** from NumPy. log1p is just going to be log, except you add one because you can't take the log of zero, so in case you have zero in your data set. 
[I used log to transform data in this Lab](https://github.com/sundas586/Data_Cleaning/blob/main/DataCleaning_IBM_DataScience.py)
- **boxcox** is just a more complex way to find the ideal way to transform from a skewed data set to a normal distribution.
-  freom sklearn.preprocessing import **polynomial Features**
  
![a](https://user-images.githubusercontent.com/33677647/201416157-c261e892-7fd3-4261-b267-b4d29d4a1c32.JPG)

Such as a larger box office probably will not have a linear relationship with the budget, but rather it would have diminishing returns. So larger the budget, after some time, you probably won't get as much return on your box office. So that isn't a linear relationship, but it may be linear with the relationship of log(x).

![a](https://user-images.githubusercontent.com/33677647/201421729-e54c1e7a-6667-4fc3-8efe-c8f0ee97115b.JPG)

![Capture](https://user-images.githubusercontent.com/33677647/201498167-45c9bc6b-e745-4e73-b6ee-efbd42094d29.JPG) 
 
(Means box office and budget had no linear/ positive or negitive co-linearity together but we applied log function to transform the distribution, and now they have a relation togetther by which we understand that higher budget will not increase population on box office by some time).

we can estimate higher-order relationships in this data by adding **polynomial features**.
So instead of only incorporating budget, we can incorporate budget squared as a feature and add more flexibility to our model and fit a linear model using these polynomial features. 

So just rather than straight, it kind of curves off. Then you would have something like x squared raising the degree to two.
Play video starting at and follow transcript

![b](https://user-images.githubusercontent.com/33677647/201421762-cb338716-5be8-4305-8d6c-e119749f37f1.JPG)

We can even continue to extend this to three, or four, or any higher-order polynomial.
Again, we're changing our features, but maintaining a linear model

![c](https://user-images.githubusercontent.com/33677647/201441243-59c93a41-c49e-4f06-8372-7dffc7ec9075.JPG)

And what we see in the graph would represent not just diminishing returns, but now we have two inflection points, right? It curves up, then it curves down. And that would be the idea that after some time not only is there diminishing returns, but a higher budget may lead to less box office sales at a certain threshold.
 
 ![d](https://user-images.githubusercontent.com/33677647/201441281-0bf7c464-5e44-4e26-ba37-26a0b3d4f0f1.JPG)
 
 (What I am getting here is that when we transform the data to a higher order, it is revealed that this data is telling us that after a certain time, not only the box office will get less customers but even a huge decline will becaused even after high budgets )
 
![e](https://user-images.githubusercontent.com/33677647/201441909-66b9019b-2672-4c9d-8e50-0a684618947e.JPG)


## 2_Feature Encoding

Encoding, converting non-numeric features such as categorical or ordinal features to numeric features.

- Nominal data : values with no order, such as red/blue/green, married/ unmarried, true/ false.
- Ordinal data : values with order, such as high/medium/low, cold/warm/hot.

In case of nominal data, If we have only two values i.e male/ female, pass/fail, married/unmarried, then we can do **binary encoding** i.e directly replace them by 1,0.
But if we have multiple values we do **One-Hot Encoding**. Which creates a new column for eac value, so we have several new columns.(I personally prefer getdummies then one hot encoding).

In case of Ordinal data, if w have 3 values, good/better/best, or low/average/high, we can simply replace the by 1/2/3.

## Feature Scaling

Adjusting of variables scale to allow comparison of variables with different scales.  

Tip--->In linear algebra, I studied it as "Changing of basis", means if one person is using some units, and you are using different units for same work, then you convert your units to that persons units to understand his work as well.

For plots I take it as if column-1 has data in mm and column-2 has data in Km then we convert column one data from mm to km so that the plotting can be correct, and the results

Different continuous features will often have different scales in a real-world data. 

![ad](https://user-images.githubusercontent.com/33677647/201444355-fb93cbaf-8445-449c-95df-07d9b5ca2c23.JPG)

Normalization means the area of plot equals 1, but scaling just to get all the columns on single/same unit, it deletes the mean from the data, and now our mean equals zero! and rest is also the data - mean.

![a](https://user-images.githubusercontent.com/33677647/201444888-1cfe1d5a-d8ba-4668-9032-78be90109524.JPG)
![b](https://user-images.githubusercontent.com/33677647/201444896-f99c8763-fb87-449b-a791-4a20f10b5cb9.JPG)

==========================================================================================

![c](https://user-images.githubusercontent.com/33677647/201444908-0a290e29-50f9-4ae7-85de-82f3b82e9601.JPG)
![d](https://user-images.githubusercontent.com/33677647/201444914-84fcbb4e-d0d2-4dfd-af9b-a8d2cf1dc605.JPG)

==========================================================================================

![e](https://user-images.githubusercontent.com/33677647/201444954-f75f7c2f-bfa8-4ce5-984a-ae9fadd14fab.JPG)
![f](https://user-images.githubusercontent.com/33677647/201444962-8886b03f-b4c8-4afd-87df-9668d6f5cafd.JPG)







 
 
