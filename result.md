Color features shape: (25959, 128)
HOG features shape: (25959, 6561)
LBP features shape: (25959, 38)
Merged features shape: (25959, 6727)
Running Knn cross validation:
Evaluating KNN with k=4
KNN HOG Accuracy (k=4): 0.6919
KNN LBP Accuracy (k=4): 0.6196
KNN HSV Accuracy (k=4): 0.5889
KNN PCA HOG Accuracy (k=4): 0.7318
KNN HOG + LBP + HSV Accuracy (k=4): 0.7091
KNN PCA HOG + LBP + HSV Accuracy (k=4): 0.7351
Evaluating KNN with k=6
KNN HOG Accuracy (k=6): 0.6863
KNN LBP Accuracy (k=6): 0.6344
KNN HSV Accuracy (k=6): 0.5936
KNN PCA HOG Accuracy (k=6): 0.7402
KNN HOG + LBP + HSV Accuracy (k=6): 0.7048
KNN PCA HOG + LBP + HSV Accuracy (k=6): 0.7486
Evaluating KNN with k=8
KNN HOG Accuracy (k=8): 0.6775
KNN LBP Accuracy (k=8): 0.6386
KNN HSV Accuracy (k=8): 0.5931
KNN PCA HOG Accuracy (k=8): 0.7451
KNN HOG + LBP + HSV Accuracy (k=8): 0.7005
KNN PCA HOG + LBP + HSV Accuracy (k=8): 0.7546
Evaluating KNN with k=10
KNN HOG Accuracy (k=10): 0.6713
KNN LBP Accuracy (k=10): 0.6467
KNN HSV Accuracy (k=10): 0.5984
KNN PCA HOG Accuracy (k=10): 0.7441
KNN HOG + LBP + HSV Accuracy (k=10): 0.6944
KNN PCA HOG + LBP + HSV Accuracy (k=10): 0.7565
Evaluating KNN with k=12
KNN HOG Accuracy (k=12): 0.6640
KNN LBP Accuracy (k=12): 0.6485
KNN HSV Accuracy (k=12): 0.5976
KNN PCA HOG Accuracy (k=12): 0.7444
KNN HOG + LBP + HSV Accuracy (k=12): 0.6906
KNN PCA HOG + LBP + HSV Accuracy (k=12): 0.7568
Evaluating KNN with k=16
KNN HOG Accuracy (k=16): 0.6540
KNN LBP Accuracy (k=16): 0.6511
KNN HSV Accuracy (k=16): 0.6015
KNN PCA HOG Accuracy (k=16): 0.7438
KNN HOG + LBP + HSV Accuracy (k=16): 0.6819
KNN PCA HOG + LBP + HSV Accuracy (k=16): 0.7564
Evaluating KNN with k=24
KNN HOG Accuracy (k=24): 0.6408
KNN LBP Accuracy (k=24): 0.6528
KNN HSV Accuracy (k=24): 0.6035
KNN PCA HOG Accuracy (k=24): 0.7405
KNN HOG + LBP + HSV Accuracy (k=24): 0.6667
KNN PCA HOG + LBP + HSV Accuracy (k=24): 0.7542
Evaluating KNN with k=32
KNN HOG Accuracy (k=32): 0.6300
KNN LBP Accuracy (k=32): 0.6561
KNN HSV Accuracy (k=32): 0.6060
KNN PCA HOG Accuracy (k=32): 0.7379
KNN HOG + LBP + HSV Accuracy (k=32): 0.6575
KNN PCA HOG + LBP + HSV Accuracy (k=32): 0.7514
Evaluating KNN with k=60
KNN HOG Accuracy (k=60): 0.6116
KNN LBP Accuracy (k=60): 0.6615
KNN HSV Accuracy (k=60): 0.6088
KNN PCA HOG Accuracy (k=60): 0.7243
KNN HOG + LBP + HSV Accuracy (k=60): 0.6391
KNN PCA HOG + LBP + HSV Accuracy (k=60): 0.7419
KNN cross-validation completed.
Starting SVC cross-validation:
Running CV for all SVC pipelines (for box plot)...
SVC HOG Accuracy: 0.8270 +/- 0.0032
SVC LBP Accuracy: 0.6820 +/- 0.0046
SVC HSV Accuracy: 0.6351 +/- 0.0044
SVC PCA HOG Accuracy: 0.7865 +/- 0.0048
SVC HOG + LBP + HSV Accuracy: 0.8405 +/- 0.0061
SVC PCA HOG + LBP + HSV Accuracy: 0.8067 +/- 0.0021
Generating SVC comparison box plot...
c:\Users\DiegoPC\Documents\Pattern\DogsVsCats\DogVsCat.py:264: MatplotlibDeprecationWarning: The 'labels' parameter of boxplot() has been renamed 'tick_labels' since Matplotlib 3.9; support for the old name will be dropped in 3.11.
  plt.boxplot(data_to_plot, labels=labels_to_plot)
running SVC on PCA HOG + LBP + HSV features
SVC PCA HOG + LBP + HSV Accuracy: 0.8071

--- Classification Report ---
              precision    recall  f1-score   support

     Cat (0)       0.81      0.81      0.81     12990
     Dog (1)       0.81      0.81      0.81     12969

    accuracy                           0.81     25959
   macro avg       0.81      0.81      0.81     25959
weighted avg       0.81      0.81      0.81     25959

Generating Confusion Matrix...
Generating ROC Curve...
Script finished.