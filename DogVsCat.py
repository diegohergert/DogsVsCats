import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import cross_val_score, KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


"This method loads the images from the cat and dog directorys"
"This then returns the images (128x128) in color, grayscale and their labels"
def load_images(cat_dir, dog_dir, image_size=(128,128)):
    images = []
    imagesGray = []
    labels = []
    for filename in os.listdir(cat_dir):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            img = cv2.imread(os.path.join(cat_dir, filename), cv2.IMREAD_COLOR)
            if img is None:
                print(f"Warning: Skipping corrupt or unreadable file: {filename}")
                continue
            img = cv2.resize(img, image_size)
            

            images.append(img)
            imagesGray.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            labels.append(0)  # 0 for cat
    for filename in os.listdir(dog_dir):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            img = cv2.imread(os.path.join(dog_dir, filename), cv2.IMREAD_COLOR)
            if img is None:
                print(f"Warning: Skipping corrupt or unreadable file: {filename}")
                continue
            img = cv2.resize(img, image_size)

            images.append(img)
            imagesGray.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            labels.append(1)  # 1 for dog
    cv2.destroyAllWindows()
    return np.array(images), np.array(imagesGray), np.array(labels)


cat_dir = os.path.join("all_cats_merged")
dog_dir = os.path.join("all_dogs_merged")
'''
cat_dir = os.path.join("CatImages", "cats")
dog_dir = os.path.join("DogImages", "dogs")
'''

images, imagesGray, labels = load_images(cat_dir,dog_dir)
print(f"Loaded {len(images)} images.")

#HSV Color Features
colorFeatures = []
hsv_bins = [16, 8]
for img in images:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0, 1], None, hsv_bins, [0, 180, 0, 256])
    hist_h = cv2.normalize(hist_h, hist_h).flatten()
    colorFeatures.append(hist_h)
colorFeatures =  np.array(colorFeatures).squeeze()
print(f"Color features shape: {colorFeatures.shape}")

# HOG Features
winsize = (128, 128)
cellSize = (16, 16)
blockSize = (48, 48)
blockStride = (10, 10)
nbins = 9
hog = cv2.HOGDescriptor(winsize, blockSize, blockStride, cellSize, nbins)
hogfeatures = []
for img in imagesGray:
    h = hog.compute(img)
    hogfeatures.append(h)
hogfeatures = np.array(hogfeatures).squeeze()
print(f"HOG features shape: {hogfeatures.shape}")

# LBP Features
lpb_features = []
radius = 3
n_points = 36
for img in imagesGray:
    lbp = local_binary_pattern(img, n_points, radius, method='uniform')
    n_bins = int(lbp.max() + 1)
    (hist, _) = np.histogram(lbp, density= True, bins = n_bins, range=(0, n_bins))
    hist /= (hist.sum() + 1e-8) #prevent division by zero
    lpb_features.append(hist)
lpb_features = np.array(lpb_features).squeeze()
print(f"LBP features shape: {lpb_features.shape}")

# Merging Features
mergeFeatures = np.hstack((hogfeatures, lpb_features, colorFeatures))
print(f"Merged features shape: {mergeFeatures.shape}")

# Setting up cross-validation strategy
cv_strategy = KFold(n_splits=5, shuffle=True, random_state=42)

# HOG pipeline
pipe_hog = Pipeline([
    ('scaler', StandardScaler()),
    ('model', KNeighborsClassifier())
])

# LBP pipeline
pipe_lbp = Pipeline([
    ('scaler', StandardScaler()),
    ('model', KNeighborsClassifier())
])

# HSV pipeline
pipe_hsv = Pipeline([
    ('scaler', StandardScaler()),
    ('model', KNeighborsClassifier())
])

# PCA on HOG features pipeline
pipe_pca_hog = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=42)),
    ('model', KNeighborsClassifier())
])

# HOG + LBP pipeline
pipe_mix = Pipeline([
    ('scaler', StandardScaler()),
    ('model', KNeighborsClassifier())
])

# PCA HOG + LBP + hsv (cant directly use due to size of features (LBP << HOG)
hog_transformer = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=42))
])

lbp_transformer = Pipeline([
    ('scaler', StandardScaler())
])

hsv_transformer = Pipeline([
    ('scaler', StandardScaler())
])

preprocessor_mixed = ColumnTransformer(
    transformers=[
        ('hog_pca', hog_transformer, list(range(hogfeatures.shape[1]))),
        ('lbp', lbp_transformer, list(range(hogfeatures.shape[1], hogfeatures.shape[1] + lpb_features.shape[1]))),
        ('hsv', hsv_transformer, list(range(hogfeatures.shape[1] + lpb_features.shape[1], mergeFeatures.shape[1])))
    ],
)

pipe_mix_pca = Pipeline([
    ('preprocessor', preprocessor_mixed),
    ('model', KNeighborsClassifier())
]
)



# Running KNN classifier
print("Running Knn cross validation:")
k_values = [4, 6, 8, 10, 12, 16, 24, 32, 60]
knnAccuracyH = {}
knnAccuracyL = {}
knnAccuracyHSV = {}
knnAccuracyM = {}
knnAccuracyPH = {}
knnAccuracyPM = {}

for k in k_values:
    print(f"Evaluating KNN with k={k}")
    pipe_hog.set_params(model__n_neighbors=k)
    pipe_lbp.set_params(model__n_neighbors=k)
    pipe_hsv.set_params(model__n_neighbors=k)
    pipe_pca_hog.set_params(model__n_neighbors=k)
    pipe_mix.set_params(model__n_neighbors=k)
    pipe_mix_pca.set_params(model__n_neighbors=k)

    scores_h = cross_val_score(pipe_hog, hogfeatures, labels, cv=cv_strategy, scoring='accuracy', n_jobs=-1)
    scores_l = cross_val_score(pipe_lbp, lpb_features, labels, cv=cv_strategy, scoring='accuracy', n_jobs=-1)
    scores_hsv = cross_val_score(pipe_hsv, colorFeatures, labels, cv=cv_strategy, scoring='accuracy', n_jobs=-1)
    scores_ph = cross_val_score(pipe_pca_hog, hogfeatures, labels, cv=cv_strategy, scoring='accuracy', n_jobs=-1)
    scores_m = cross_val_score(pipe_mix, mergeFeatures, labels, cv=cv_strategy, scoring='accuracy', n_jobs=-1)
    scores_pm = cross_val_score(pipe_mix_pca, mergeFeatures, labels, cv=cv_strategy, scoring='accuracy', n_jobs=-1)

    knnAccuracyH[k] = scores_h.mean()
    knnAccuracyL[k] = scores_l.mean()
    knnAccuracyHSV[k] = scores_hsv.mean()
    knnAccuracyPH[k] = scores_ph.mean()
    knnAccuracyM[k] = scores_m.mean()
    knnAccuracyPM[k] = scores_pm.mean()

    print(f"KNN HOG Accuracy (k={k}): {knnAccuracyH[k]:.4f}")
    print(f"KNN LBP Accuracy (k={k}): {knnAccuracyL[k]:.4f}")
    print(f"KNN HSV Accuracy (k={k}): {knnAccuracyHSV[k]:.4f}")
    print(f"KNN PCA HOG Accuracy (k={k}): {knnAccuracyPH[k]:.4f}")
    print(f"KNN HOG + LBP + HSV Accuracy (k={k}): {knnAccuracyM[k]:.4f}")
    print(f"KNN PCA HOG + LBP + HSV Accuracy (k={k}): {knnAccuracyPM[k]:.4f}")
print("KNN cross-validation completed.")

plt.figure(figsize=(10, 6))
plt.plot(list(knnAccuracyH.keys()), list(knnAccuracyH.values()), marker='o', label='KNN HOG')
plt.plot(list(knnAccuracyL.keys()), list(knnAccuracyL.values()), marker='^', label='KNN LBP')
plt.plot(list(knnAccuracyHSV.keys()), list(knnAccuracyHSV.values()), marker='s', label='KNN HSV')
plt.plot(list(knnAccuracyPH.keys()), list(knnAccuracyPH.values()), marker='+', label='KNN PCA HOG')
plt.plot(list(knnAccuracyM.keys()), list(knnAccuracyM.values()), marker='x', label='KNN HOG + LBP + HSV')
plt.plot(list(knnAccuracyPM.keys()), list(knnAccuracyPM.values()), marker='*', label='KNN PCA HOG + LBP + HSV')
plt.title('KNN Classifier Accuracy for Different k Values')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig("results/KNN_Accuracy_vs_k.png")
plt.close()

print("Starting SVC cross-validation:")
#best params I found via many trials
svc_model = SVC(kernel='rbf', C=2, gamma='scale', probability=True)

pipe_hog.set_params(model=svc_model)
pipe_lbp.set_params(model=svc_model)
pipe_hsv.set_params(model=svc_model)
pipe_pca_hog.set_params(model=svc_model)
pipe_mix.set_params(model=svc_model)
pipe_mix_pca.set_params(model=svc_model)

print("Running CV for all SVC pipelines (for box plot)...")
scores_svc_hog = cross_val_score(pipe_hog, hogfeatures, labels, cv=cv_strategy, scoring='accuracy', n_jobs=-1)
scores_svc_lbp = cross_val_score(pipe_lbp, lpb_features, labels, cv=cv_strategy, scoring='accuracy', n_jobs=-1)
scores_svc_hsv = cross_val_score(pipe_hsv, colorFeatures, labels, cv=cv_strategy, scoring='accuracy', n_jobs=-1)
scores_svc_pca_hog = cross_val_score(pipe_pca_hog, hogfeatures, labels, cv=cv_strategy, scoring='accuracy', n_jobs=-1)
scores_svc_mix = cross_val_score(pipe_mix, mergeFeatures, labels, cv=cv_strategy, scoring='accuracy', n_jobs=-1)
scores_svc_mix_pca = cross_val_score(pipe_mix_pca, mergeFeatures, labels, cv=cv_strategy, scoring='accuracy', n_jobs=-1)

# Print the mean and standard deviation for comparison
print(f"SVC HOG Accuracy: {scores_svc_hog.mean():.4f} +/- {scores_svc_hog.std():.4f}")
print(f"SVC LBP Accuracy: {scores_svc_lbp.mean():.4f} +/- {scores_svc_lbp.std():.4f}")
print(f"SVC HSV Accuracy: {scores_svc_hsv.mean():.4f} +/- {scores_svc_hsv.std():.4f}")
print(f"SVC PCA HOG Accuracy: {scores_svc_pca_hog.mean():.4f} +/- {scores_svc_pca_hog.std():.4f}")
print(f"SVC HOG + LBP + HSV Accuracy: {scores_svc_mix.mean():.4f} +/- {scores_svc_mix.std():.4f}")
print(f"SVC PCA HOG + LBP + HSV Accuracy: {scores_svc_mix_pca.mean():.4f} +/- {scores_svc_mix_pca.std():.4f}")


print("Creating SVC comparison box plot...")
data_to_plot = [
    scores_svc_hog,
    scores_svc_lbp,
    scores_svc_hsv,
    scores_svc_pca_hog,
    scores_svc_mix,
    scores_svc_mix_pca
]
labels_to_plot = [
    'SVC HOG',
    'SVC LBP',
    'SVC HSV',
    'SVC PCA HOG',
    'SVC Mixed',
    'SVC Mixed (PCA)'
]

plt.figure(figsize=(12, 8))
plt.boxplot(data_to_plot, labels=labels_to_plot)
plt.title('SVC Model Performance Comparison (5-Fold CV)')
plt.ylabel('Accuracy')
plt.xticks(rotation=30, ha='right') # Rotate labels to prevent overlap
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.tight_layout() # Adjust plot to fit labels
plt.savefig("results/SVC_Model_Comparison_Boxplot.png")
plt.close()



print("running SVC on PCA HOG + LBP + HSV features")
scores_svc_mix_pca = cross_val_score(pipe_mix_pca, mergeFeatures, labels, cv=cv_strategy, scoring='accuracy', n_jobs=-1)
print(f"SVC PCA HOG + LBP + HSV Accuracy: {scores_svc_mix_pca.mean():.4f}")
y_pred_cv = cross_val_predict(pipe_mix_pca, mergeFeatures, labels, cv=cv_strategy, n_jobs=-1)
y_probs_cv = cross_val_predict(pipe_mix_pca, mergeFeatures, labels, cv=cv_strategy, method='predict_proba', n_jobs=-1)
# Get the probabilities for the "positive" class (Dog, label 1)
y_probs_cv_dog = y_probs_cv[:, 1]


# 1. Classification Report (Precision, Recall, F1-Score)
print("\n--- Classification Report ---")
print(classification_report(labels, y_pred_cv, target_names=["Cat (0)", "Dog (1)"]))


# 2. Confusion Matrix
print("Generating Confusion Matrix...")
cm = confusion_matrix(labels, y_pred_cv)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Cat", "Dog"])
disp.plot()
plt.title("SVC (PCA+LBP + HSV) CV Confusion Matrix")
plt.savefig("results/SVC_PCA_LBP_HSV_Confusion_Matrix.png")
plt.close()


# ROC Curve and AUC Score
print("Generating ROC Curve...")
fpr, tpr, thresholds = roc_curve(labels, y_probs_cv_dog)
auc_score = roc_auc_score(labels, y_probs_cv_dog)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'SVC (AUC = {auc_score:.4f})')
plt.plot([0, 1], [0, 1], 'k--') # Dashed diagonal line for random chance
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for SVC (PCA+LBP + HSV) from Cross-Validation')
plt.legend(loc='lower right')
plt.grid(True)
plt.savefig("results/SVC_PCA_LBP_HSV_ROC_Curve.png")
plt.close()

print("Script finished.")