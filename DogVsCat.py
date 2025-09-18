import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

"load images from directory and resize them"
def load_images(cat_dir, dog_dir, image_size=(128,128)):
    images = []
    labels = []
    for filename in os.listdir(cat_dir):
        if filename.endswith('.jpg'):
            img = cv2.imread(os.path.join(cat_dir, filename))
           
            img = cv2.resize(img, image_size)
            
            cv2.imshow(f'Image Loading {filename}', img)
            print(f'Image Loading {filename}')
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            images.append(img)
            labels.append(0)  # 0 for cat
    for filename in os.listdir(dog_dir):
        if filename.endswith('.jpg'):
            img = cv2.imread(os.path.join(dog_dir, filename))
            img = cv2.resize(img, image_size)

            cv2.imshow(f'Image Loading {filename}', img)
            print(f'Image Loading {filename}')
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            images.append(img)
            labels.append(1)  # 1 for dog
    return np.array(images), np.array(labels)



def extract_hog(images, image_size=(128,128)):
    hog = cv2.HOGDescriptor()
    features = []
    for img in images:
        img = cv2.resize(img, image_size)
        h = hog.compute(img)
        features.append(h.flatten())
    return np.array(features)


cat_dir = 'CatImages\cats'
dog_dir = 'DogImages\dogs'
images, labels = load_images(cat_dir,dog_dir)
print(f"Loaded {len(images)} images.")

features = extract_hog(images)
print(f"Extracted features of shape: {features.shape}")

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")



