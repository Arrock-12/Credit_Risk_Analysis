# Credit_Risk_Analysis
# Overview
Using a loan application dataset with a multitude of features, we trained and evaluated machine learning models to assess loan risks. We used imbalancd-learn and scikit-learn libraries to create various models because of the severe imbalance between low-risk loans (68,470 applications) vs high-risk (347).  
We created models where we oversampled using RandomOverSampler and SMOTE algorithms, undersampled using ClusterCentroids algorithm, and a combination using SMOTEEN. We then compared BalancedRandomForestClassifier and EasyEnembleClassifier machine learning models to reduce bias and predict loan risk. 
### Resources
Dataset:
-	LoanStats_2019Q1.csv

Applications/Technologies
-	Jupyter Notebook
-	Imbalanced-learn and sci-kit learn libraries for Pandas

## Results
To compare the results of the machine learning models, we set the random_state for each model to 1. This allows the results to be replicated and allows for a cross-model comparison of results.

The results from the learning models are as follows:

Naïve Random Oversampling using RandomOverSampler algorithm from imbalance-learn library:

![Naive_Random_imbalanced_report](https://user-images.githubusercontent.com/101822948/183743334-e26fa134-1b81-4ca8-9d84-1df8b8d62852.png)

-	Balanced Accuracy Score: 63.8%
-	Precision score:
    - High-risk: .01
    - Low-risk: 1
    -	Average: .99
-	Recall
    -	High-risk: .62
    -	Low-risk: .66
    -	Average: .65

SMOTE Oversampling using SMOTE algorithm from imbalance-learn library

![SMOTE_imbalanced_classification_report](https://user-images.githubusercontent.com/101822948/183743646-de80e448-405d-482d-b7db-e00e4fca4ead.png)

-	Balanced Accuracy Score: 62.4%
-	Precision score:
    -	High-risk: .01
    -	Low-risk: 1
    -	Average: .99
-	Recall
    -	High-risk: .62
    -	Low-risk: .63
    -	Average: .63

Undersampling using ClusterCentroids algorithm from scikit-learn library:

![undersampled_imbalanced_classification_report](https://user-images.githubusercontent.com/101822948/183743751-c3ace211-30c4-4b8f-8f94-ba7ba4f1998a.png)

-	Balanced Accuracy Score: 51.6%
-	Precision score:
    -	High-risk: .01
    -	Low-risk: 1
    -	Average: .99
-	Recall
    -	High-risk: .60
    -	Low-risk: .43
    -	Average: .44

Combination using SMOTEEN algorithm from imbalance-learn library:

![SMOTEEN_imbalanced_classification_report](https://user-images.githubusercontent.com/101822948/183744747-f38adb97-06ac-4a34-8ec4-0e67994c8bef.png)

 -	Balanced Accuracy Score: 62.8%
 -	Precision score:
    -	High-risk: .01
    -	Low-risk: 1
    -	Average: .99
-	Recall
    -	High-risk: .66
    -	Low-risk: .60
    -	Average: .60

Ensemble Learner – Balanced Random Forest using RandomForestClassifier algorithm from scikit-learn library:

![balanced_forest_imbalanced_classification_report](https://user-images.githubusercontent.com/101822948/183744801-e0cef3dd-9bbd-4aa3-b271-030f6c35b4bf.png)

-	Balanced Accuracy Score: 65.5%
-	Precision score:
    -	High-risk: .07
    -	Low-risk: 1
    -	Average: .99
-	Recall
    -	High-risk: .91
    -	Low-risk: .94
    -	Average: .94

Ensemble Learner – AdaBoost Classifier using EasyEnsembleClassifier from imbalanced learn library:

![Adaboost_imbalanced_report](https://user-images.githubusercontent.com/101822948/183744835-92c285e9-eefa-477f-a173-19dedec182a1.png)

-	Balanced Accuracy Score: 92.5%
-	Precision score:
    -	High-risk: .07
    -	Low-risk: 1
    -	Average: .99
-	Recall
    -	High-risk: .91
    -	Low-risk: .94
    -	Average: .94

## Summary
Overall, both the ensemble learning methods produced better accuracy and recall scores compared to the resampling techniques. The precision scores for ALL tests were all very similar with high-risk application applications scoring between .01 and .07 while the precision score for the low-risk scores were all 1. This makes sense because of the severe imbalanced between low-risk applications and high-risk applications. This indicates that all tests accurately predicted low-risk applications, but did poorly predicting high-risk scores. The f1 scores for high-risk loans were all consistently very low (.01 or .02 for the resampling models and .14 for both of the ensemble learning methods.) 

Recall scores for the resampling tests for both high- and low-risk loans were in the low to mid 60% range (.62 to .64 mostly, although the undersampling model produced the worst recall scores with 60% for high-risk loans and 43% for low-risk loans). The recall scores for the ensemble learning models were notably higher in the low to mid 90% range. 

Based on the results from this analysis, either of the ensemble learning methods clearly outperformed the resampling methods. This is expected since both of the ensemble learning methods use multiple weak learners to create a smarter model for more accurate predictions.  
