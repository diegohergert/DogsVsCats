import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


"load images from directory and resize them"
def load_images(cat_dir, dog_dir, image_size=(128,128)):
    images = []
    imagesGray = []
    labels = []
    cv2.namedWindow('Image Loading', cv2.WINDOW_NORMAL)
    for filename in os.listdir(cat_dir):
        if filename.endswith('.jpg'):
            img = cv2.imread(os.path.join(cat_dir, filename))
            img = cv2.resize(img, image_size)
            
            cv2.imshow('Image Loading', img)
            print(f'Image Loading {filename}')

            images.append(img)
            imagesGray.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            labels.append(0)  # 0 for cat
    for filename in os.listdir(dog_dir):
        if filename.endswith('.jpg'):
            img = cv2.imread(os.path.join(dog_dir, filename))
            img = cv2.resize(img, image_size)


            cv2.imshow('Image Loading', img)
            print(f'Image Loading {filename}')

            images.append(img)
            imagesGray.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            labels.append(1)  # 1 for dog
    cv2.destroyAllWindows()
    return np.array(images), np.array(imagesGray), np.array(labels)



cat_dir = 'CatImages\cats'
dog_dir = 'DogImages\dogs'
images, imagesGray, labels = load_images(cat_dir,dog_dir)
print(f"Loaded {len(images)} images.")

hog = cv2.HOGDescriptor()
hogfeatures = []
for img in imagesGray:
    h = hog.compute(img)
    hogfeatures.append(h)

sift = cv2.SIFT_create(500)
# 500 per image
siftfeatures = []
for img in imagesGray:
    keypoints, des = sift.detectAndCompute(img, None)
    if des is not None:
        descriptors = des.flatten()
        needed_size = 128 * 128  # Example fixed size
        if descriptors.size < needed_size:
            descriptors = np.pad(descriptors, (0, needed_size - descriptors.size), 'constant')
        else:
            descriptors = descriptors[:needed_size]
    else:
        descriptors = np.zeros(128 * 128)  # If no keypoints are found
    siftfeatures.append(descriptors)

print(f"SIFT descriptors shape for first image: {descriptors.shape}")
print(f"Extracted features of shape: {len(hogfeatures)}")

X_train, X_test, y_train, y_test = train_test_split(hogfeatures, labels, test_size=0.2, random_state=42)
X_trainGray, X_testGray, y_trainGray, y_testGray = train_test_split(siftfeatures, labels, test_size=0.2, random_state=42)

print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

knnAccuracy = {}
for i in range(1, 40, 5):
    model = KNeighborsClassifier(n_neighbors=i)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.2f}")
    knnAccuracy[i] = accuracy

plt.plot(list(knnAccuracy.keys()), list(knnAccuracy.values()), marker='o')
plt.title('KNN Classifier Accuracy over Iterations - Hog')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.grid()
plt.show()

knnAccuracy = {}
for i in range(1, 40, 5):
    model = KNeighborsClassifier(n_neighbors=i)
    model.fit(X_trainGray, y_trainGray)

    y_pred = model.predict(X_testGray)
    accuracy = accuracy_score(y_testGray, y_pred)
    print(f"Test Accuracy: {accuracy:.2f}")
    knnAccuracy[i] = accuracy

plt.plot(list(knnAccuracy.keys()), list(knnAccuracy.values()), marker='o')
plt.title('KNN Classifier Accuracy over Iterations - SIFT')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.grid()
plt.show()


plt.close()