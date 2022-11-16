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
- Mutual Information 


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
- from sklearn.preprocessing import **polynomial Features**

![log1p](https://user-images.githubusercontent.com/33677647/201547702-7d0e1967-243f-4940-b0e0-01461186d447.JPG)

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

## 3_Feature Scaling

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

## 4_Mutual Information 

[MI_Kaggle](https://www.kaggle.com/code/ryanholbrook/mutual-information)
[MI_Towards_DataScience](https://towardsdatascience.com/select-features-for-machine-learning-model-with-mutual-information-534fe387d5c8)

**What is mutual information (MI) in feature selection?**
Between two random variables, one taken from any feature and one taken from target feature, there is a non-negative value, which measures the dependency between the variables. It is equal to zero if and only if two random variables are totally independent of each other, and higher values mean higher dependency.

MI is widely used in machine learning to measure dependency among different features in data.

 If you knew the value of a feature, how much more confident would you be about the target?
 
![Capture](https://user-images.githubusercontent.com/33677647/201500523-32caed17-1523-451d-a639-12a6ea9b0125.JPG)
![a](https://user-images.githubusercontent.com/33677647/201546277-652a5a56-0eb0-407c-a57b-83712423ae92.JPG)
![b](https://user-images.githubusercontent.com/33677647/201546279-d9a79a05-f525-4bf1-8de0-fc74997fcbc7.JPG)

Here's an example from the Ames Housing data. The figure shows the relationship between the exterior quality of a house and the price it sold for. Each point represents a house.

![Capture](https://user-images.githubusercontent.com/33677647/201501255-926d8936-56ec-46f1-ad80-b6eb80f54ea0.JPG)

So the feature "exterior quality" decreased the uncertinity for price feature.

Here uncertainty  known as **"entropy"**. The entropy of a variable means roughly: "how many yes-or-no questions you would need to describe an occurance of that target variable, on average." The more questions you have to ask, the more uncertain you must be about the variable.

Where as **Mutual information** is how many questions you expect the  from the "exterior quality" feature to answer about the target feature.

The more the MI the less the entropy.

The more the MI (stronge rel. btw both features, so to get most info from "exterior quality" about target) the less the entropy (uncertinity about what will be the target result).

The least possible mutual information between quantities is 0.0. When MI is zero, the quantities are independent: neither can tell you anything about the other. 
utual information is a logarithmic quantity, so it increases very slowly.)

The next figure will give you an idea of how MI values correspond to the kind and degree of association a feature has with the target.
 
 ![Capture](https://user-images.githubusercontent.com/33677647/201501707-3a1efe59-70d7-4807-b7c2-2bc17c243dcd.JPG)
![Capture2](https://user-images.githubusercontent.com/33677647/201501708-61923b6c-bae6-49d8-ad68-d5ea5954d6e4.JPG)

![a](https://user-images.githubusercontent.com/33677647/201547514-5535a16f-4ef0-48fd-a8dc-f5a2ed4efef1.JPG)
![b](https://user-images.githubusercontent.com/33677647/201547516-015fee08-92d0-4705-8925-139d740953b5.JPG)

## 5_ K-Means Clustering
 
Unsupervised algorithms don't make use of a target, instead their purpose is to learn some property of the data, and in order to understand the underline structure of After dividing our dataset in two clustersour data set, we divide data into clusters to make predictions out of it.<br/>
Clustering simply means the assigning of data points to groups based upon how similar the points are to each other 
 
 

![a](https://user-images.githubusercontent.com/33677647/201707914-a00f13a0-7cdd-445c-8563-e5b6ed13e04a.JPG)

as we donot have any target variable here, so in order to identify our data set we divide it into **"K"** clusters,
Here k is a free parameter, means we can initialize k by our selves.
Here k = 2

![b](https://user-images.githubusercontent.com/33677647/201707922-e437ef3c-e95a-449e-9ee6-d3ae796e32c7.JPG)

So we we tell our model to set k = 2 , it will put 2 centroids randomly any where on our data set and

![aa](https://user-images.githubusercontent.com/33677647/201709062-a4abf3b3-1279-4f07-a851-8e9a999c1486.JPG)

then calculate and compare the distance of each data point to both centroids using euclidean distance formula, then all data points are clustered to any one of the centroids which is more close to them. 

![b](https://user-images.githubusercontent.com/33677647/201708780-aed3706e-a9cb-4c56-b2c3-a53268d65ff4.JPG) <br/><br/>
 <br/>
After dividing our dataset in two clusters, the center of each cluster is calculated, and then the centroids are placed on those centers.

**ALERT!!!**<br/>
If features of your data do not have same scales, model will to find the center of a cluster correct,<br/>
so if the data features have different scales, please scale them before putting data to the model
 
![a](https://user-images.githubusercontent.com/33677647/201710363-c9acc86f-7ad6-4864-9773-a42d650637be.JPG)
![b](https://user-images.githubusercontent.com/33677647/201710384-290c94f3-fd62-4905-a655-fd5c12a575e4.JPG)

 After placing the centroids in the center of each cluster, <br/>
 We **recompute** the distance of each data point to 2 centroids,<br/>
 repeat the process place the centroid again to the center of its nearest datapoints.<br/>
 
 ![a](https://user-images.githubusercontent.com/33677647/201712483-6e701b48-38cd-4a4e-8638-72f8368b35a3.JPG)
 
 The datapoints more near to red are now the partof red cluster.
 
![a](https://user-images.githubusercontent.com/33677647/201714112-9ac539c6-f1ec-4a34-8a4b-748a74d28db5.JPG)
![a](https://user-images.githubusercontent.com/33677647/201714702-605f627e-f2c9-46ba-949f-d161416d6e1d.JPG)

Keep repeating the process of calculating the center for centroids and then the distance of each data point to new position of centroids untill none of the point changethe cluster.
 
![a](https://user-images.githubusercontent.com/33677647/201715041-4e137d19-4487-4749-ad99-482049e76699.JPG)
 
These are now the final clusters
 
![b](https://user-images.githubusercontent.com/33677647/201715079-c2e24c8d-e032-4230-8480-403777df3857.JPG
 
![a](https://user-images.githubusercontent.com/33677647/201715634-89c1e948-4f5b-4240-9c66-1b8bd73deff1.JPG)

**QUESTION** : Now the problem is that how to select the correct k, because in real life there are many features, and its difficult to visualize them on a scatter plot.<br/>
 
 **ANSWER** : "ALGO METHOD" is a method to select the correct number of K for clustering.
 
### Algo method, technique is same as calculating the best fitted line in linear regeression, which is done by taking several **sum of squared errors (SSE)** and assuming that the best fitted line is the one that has least SSE.
 
Similarly, in K-means algorithm, the algo method takes the distance of each data point in cluster-1 to its centroid C1 and distance the data points of cluster-2 to  ,  its centroid c2 , <br/>

* all distances of cluster-1 are squared and then sumed up = S1<br/>
* all distances of cluster-1 are squared and then sumed up = S2 <br/>
 
SSE_1 = S1 + S2 

After calculating first SSE we take 3 centroids then 4 then 4,5,6,7,8,9,10,11. keep taking untill all data points become a centroid its self so there is no SSE
 
SS_2 = ...
SS_3 = ... 
.
.
.
SS_11 = ..
 
After getting the final SSE. we plot a graph of SSE<br/>
In that graph we find **Elbow Point** in our graph, similar to elbow point of best fiited line in regression analysis.<br/>
The point where we see our Elbow is our choozen **K**
 
![a](https://user-images.githubusercontent.com/33677647/201735535-14fcd946-df09-41d6-9273-c426432afe2c.JPG)
![b](https://user-images.githubusercontent.com/33677647/201735559-c80c1414-e9e6-4bd5-9cc7-bd299dbfc2f3.JPG)
 
 
You could imagine each centroid capturing points through a sequence of radiating circles. When sets of circles from competing centroids overlap they form a line. The result is what's called a **Voronoi tessallation**. The tessallation shows you to what clusters future data will be assigned; the tessallation is essentially what k-means learns from its training data. 

![tBkCqXJ](https://user-images.githubusercontent.com/33677647/201751403-4e7f3f70-76d1-44fd-9e00-40da1bb4a289.gif)
 
## 6_Principle component analysis

Principal component analysis (PCA). Just like clustering is a partitioning of the dataset based on proximity,Pca focuses on variables with high variation/spread to get more and more information about data you could think of PCA as a partitioning of the variation in the data. <br/>
PCA is a great tool to help you discover important relationships in the data and can also be used to create more informative features.
 
(Technical note: PCA is typically applied to standardized data.<br/>
- With standardized data "variation" means **"correlation"**.<br/>
- With unstandardized data "variation" means **"covariance"**.<br/>
All data in this course will be standardized before applying PCA.)<br/>
 
![a](https://user-images.githubusercontent.com/33677647/202102672-70e4c536-94ed-47fd-a088-68339724324e.JPG)

You could imagine that within this data are "axes of variation" that describe the ways the abalone tend to differ from one another.
 
 ![b](https://user-images.githubusercontent.com/33677647/202103429-41546913-4aaf-4c0e-83dd-49cf8c25a065.JPG)
 
 Notice that instead of describing abalones by their 'Height' and 'Diameter', we could just as well describe them by their 'Size' and 'Shape'. This, in fact, is the whole idea of PCA: instead of describing the data with the original features, we describe it with its axes of variation. The axes of variation become the new features.

![c](https://user-images.githubusercontent.com/33677647/202103870-32497fcc-bdd1-45aa-a196-22d766988b9b.JPG)

The new features PCA constructs are actually just linear combinations (weighted sums) of the original features:

![d](https://user-images.githubusercontent.com/33677647/202104347-afa9e7da-10ac-4c5b-ade6-c123433245e4.JPG)

These new features are called the principal components of the data. The weights themselves are called loadings. There will be as many principal components as there are features in the original dataset: if we had used ten features instead of two, we would have ended up with ten components.

A component's loadings tell us what variation it expresses through signs and magnitudes:

![e](https://user-images.githubusercontent.com/33677647/202104376-5b625bc7-13e4-4ab9-8b7a-bc1b601e0049.JPG)








 
 
