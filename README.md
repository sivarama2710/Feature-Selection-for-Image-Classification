import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.datasets import cifar10
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

# Step 1: Load CIFAR-10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = preprocess_input(x_train), preprocess_input(x_test)

# Step 2: Use ResNet50 to extract features
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(32, 32, 3), pooling='avg')
feature_extractor = tf.keras.models.Model(inputs=base_model.input, outputs=base_model.output)

train_features = feature_extractor.predict(x_train, batch_size=64, verbose=1)
test_features = feature_extractor.predict(x_test, batch_size=64, verbose=1)

# Step 3: Simulate ABC by selecting top 50 PCA features (Mock ABC)
pca = PCA(n_components=50)
train_reduced = pca.fit_transform(train_features)
test_reduced = pca.transform(test_features)

# Step 4: Train classifier on selected features
clf = RandomForestClassifier(n_estimators=100, random_state=42)

start_time = time.time()
clf.fit(train_reduced, y_train.ravel())
training_time = time.time() - start_time

# Step 5: Evaluate
preds = clf.predict(test_reduced)
acc = accuracy_score(y_test, preds)

print("Accuracy with ABC-selected features:", acc)
print("Training time:", training_time, "seconds")
