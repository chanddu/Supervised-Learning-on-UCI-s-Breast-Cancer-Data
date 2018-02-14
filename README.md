# Supervised Learning on UCI's Breast Cancer Data

#### Dataset : Click [here](https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+%28original%29)

* Partitioned the dataset into training and testing parts
* Ran Decision Tree and Logistic Reression available on the Spark on the dataset

In the first attempt, use all of the attributes (features) while training and testing the model <br>

In the second attempt, perform `dimensionality reduction` and reduce the dimensions to some chosen value K. Then, performed the training and testing on these K dimensions and reported results. The details of dimensionality reduction are available here: https://spark.apache.org/docs/latest/mllib-dimensionality-reduction.html <br>

## Link to the Decision Tree Implementation : [DT](https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/167428040012665/2841578069654501/8971546509206599/latest.html) <br>

#### Learned Decision Tree:

```
Test Error = 0.05238095238095238
Learned classification tree model:
DecisionTreeModel classifier of depth 4 with 23 nodes
  If (feature 2 <= 2.0)
   If (feature 6 <= 5.0)
    If (feature 8 <= 8.0)
     If (feature 1 <= 6.0)
      Predict: 2.0
     Else (feature 1 > 6.0)
      Predict: 2.0
    Else (feature 8 > 8.0)
     Predict: 4.0
   Else (feature 6 > 5.0)
    If (feature 1 <= 1.0)
     Predict: 2.0
    Else (feature 1 > 1.0)
     Predict: 4.0
  Else (feature 2 > 2.0)
   If (feature 6 <= 1.0)
    If (feature 2 <= 3.0)
     Predict: 2.0

Confusion matrix:
132.0  3.0   
8.0    67.0  
Summary Statistics
Accuracy = 0.9476190476190476
Precision(2.0) = 0.9428571428571428
Precision(4.0) = 0.9571428571428572
Recall(2.0) = 0.9777777777777777
Recall(4.0) = 0.8933333333333333
FPR(2.0) = 0.10666666666666667
FPR(4.0) = 0.022222222222222223
F1-Score(2.0) = 0.9600000000000001
F1-Score(4.0) = 0.9241379310344828
Weighted precision: 0.9479591836734693
Weighted recall: 0.9476190476190476
Weighted F1 score: 0.947192118226601
Weighted false positive rate: 0.07650793650793651
accuracy: Double = 0.9476190476190476
labels: Array[Double] = Array(2.0, 4.0)
```
## Link to the Decision Tree Implementation with Dimensionality reduction (PCA): [DT_PCA](https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/167428040012665/2243788287462920/8971546509206599/latest.html)

#### Learned Decision Tree (With PCA):

```
Test Error = 0.02197802197802198
numClasses: Int = 5
categoricalFeaturesInfo: scala.collection.immutable.Map[Int,Int] = Map()
impurity: String = gini
maxDepth: Int = 5
maxBins: Int = 32
model: org.apache.spark.mllib.tree.model.DecisionTreeModel = DecisionTreeModel classifier of depth 5 with 33 nodes
labelAndPreds: org.apache.spark.rdd.RDD[(Double, Double)] = MapPartitionsRDD[37] at map at <console>:71
testErr: Double = 0.02197802197802198

Confusion matrix:
117.0  1.0   
3.0    61.0  
Summary Statistics
Accuracy = 0.978021978021978
Precision(2.0) = 0.975
Precision(4.0) = 0.9838709677419355
Recall(2.0) = 0.9915254237288136
Recall(4.0) = 0.953125
FPR(2.0) = 0.046875
FPR(4.0) = 0.00847457627118644
F1-Score(2.0) = 0.9831932773109243
F1-Score(4.0) = 0.9682539682539683
Weighted precision: 0.9781194611839773
Weighted recall: 0.9780219780219781
Weighted F1 score: 0.9779398939062804
Weighted false positive rate: 0.03337155429316446
accuracy: Double = 0.978021978021978
labels: Array[Double] = Array(2.0, 4.0)
```
## Link to the Logistic Regression Implementation: [LR](https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/167428040012665/2243788287462925/8971546509206599/latest.html)

Output of the model:
```
Confusion matrix:
165.0  10.0  
5.0    82.0  
Summary Statistics
Accuracy = 0.9427480916030534
Precision(2.0) = 0.9705882352941176
Precision(4.0) = 0.8913043478260869
Recall(2.0) = 0.9428571428571428
Recall(4.0) = 0.9425287356321839
FPR(2.0) = 0.05747126436781609
FPR(4.0) = 0.05714285714285714
F1-Score(2.0) = 0.9565217391304348
F1-Score(4.0) = 0.9162011173184358
Weighted precision: 0.9442611428906112
Weighted recall: 0.9427480916030535
Weighted F1 score: 0.9431328303608015
Weighted false positive rate: 0.057362213113726676
accuracy: Double = 0.9427480916030534
labels: Array[Double] = Array(2.0, 4.0)
```

## Link to Logistic Regression Implementation with PCA: [LR_PCA](https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/167428040012665/212149162646688/8971546509206599/latest.html)

Output of the model:

```
Confusion matrix:
165.0  10.0  
3.0    84.0  
Summary Statistics
Accuracy = 0.950381679389313
Precision(2.0) = 0.9821428571428571
Precision(4.0) = 0.8936170212765957
Recall(2.0) = 0.9428571428571428
Recall(4.0) = 0.9655172413793104
FPR(2.0) = 0.034482758620689655
FPR(4.0) = 0.05714285714285714
F1-Score(2.0) = 0.9620991253644315
F1-Score(4.0) = 0.9281767955801106
Weighted precision: 0.9527468734773428
Weighted recall: 0.950381679389313
Weighted F1 score: 0.9508348402833784
Weighted false positive rate: 0.042007295152859774
accuracy: Double = 0.950381679389313
labels: Array[Double] = Array(2.0, 4.0)
```

### Statistics Summary:
![](https://github.com/chanddu/Supervised-Learning-on-UCI-s-Breast-Cancer-Data/blob/master/Statistics%20Summary%20.png)
