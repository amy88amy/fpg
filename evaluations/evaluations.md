### Evaluations and Results

#### Classification

For classification problem we have used XGBoost Classifier model, which is an implementation of Gradient Boosted Trees alogirthm.
As asked in the class the model accuracy is only measured using `7200` fradulent classes, hence `FP` and `TN` are zero for all results.

1. For base model with simple features from the original data.

    |  TP  |  FP  |
    |------|------|
    | 2896 |  0   |
    |------|------|
    | FN   |  TN  |
    |------|------|
    | 4304 |  0   |


2. Simple features with SMOTE applied

    |  TP  |  FP  |
    |------|------|
    | 7079 |  0   |
    |------|------|
    | FN   |  TN  |
    |------|------|
    | 121  |  0   |


3. Simple features with Self Paced ensembles applied

    |  TP  |  FP  |
    |------|------|
    | 6995 |  0   |
    |------|------|
    | FN   |  TN  |
    |------|------|
    | 205  |  0   |


4. DeepWalk with basic features

    |  TP  |  FP  |
    |------|------|
    | 4674 |  0   |
    |------|------|
    | FN   |  TN  |
    |------|------|
    | 2526 |  0   |

5. DeepWalk with basic features + SMOTE

    |  TP  |  FP  |
    |------|------|
    | 7079 |  0   |
    |------|------|
    | FN   |  TN  |
    |------|------|
    | 121  |  0   |


* We can observe that SMOTE helps in identifying the fraudulent transactions we higher accuracy. 
* DeepWalk was able to improve the accuracy of the base model, which could siginify that the vector representations
accounted for some information gain which helps the classification model.
* Although, we were not able to observe any gain when DeepWalk is used in conjuction with the SMOTE algorithm.


#### Clustering

* In case of such use cases, validating the ground truth and generating labelled data is an expensive and time consuming process.
* If we are able to observe some patterns and cluster those with similar patterns together then we can reduce the overhead of obtaining the ground truth.
* Also, in some cases ground truth can be acquired only after few months of investigation. We cannot wait that long to evaluate our model in the production scenario.
* We observed that `~15` clusters should give us maximum gain in information. 

    [![elbow-curve](https://github.com/PratikBarhate/fpg/blob/master/code/notebooks/kmeans_elbow.jpg)]

* As seen from the below result of K-Means clustering for `k=15`. The `two` clusters at the top contain all the fraudulent transactions.
 
    [![kmeans-15](https://github.com/PratikBarhate/fpg/blob/master/code/notebooks/kmeans_15.jpg)]

* Only `1.42%` valid transactions out of the total data fell into one of these 2 clusters


