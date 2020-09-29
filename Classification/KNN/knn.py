import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing, metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


# load data from csv
df = pd.read_csv('iris.csv')
print(df.head(3))

# data visualization
print(df['species'].value_counts())

X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values
Y = df['species'].values


# normalize data
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=42)
# 42 is the seed for the random state
print('Train set:', X_train.shape, y_train.shape)
print('Test set:', X_test.shape, y_test.shape)

# classification Classification
k = 4
# Train the model and Predict
neigh = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)

y_pred = neigh.predict(X_test)
# print(y_pred[: 5])

# accuracy evaluation
print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, y_pred))


# check for optimal k
Ks = 10
# generate arrays of zeros
mean_acc = np.zeros((Ks - 1))

for n in range(1, Ks):
    # Train Model and Predict
    neigh = KNeighborsClassifier(n_neighbors=n).fit(X_train, y_train)
    y_pred = neigh.predict(X_test)
    mean_acc[n - 1] = metrics.accuracy_score(y_test, y_pred)

plt.plot(range(1, Ks), mean_acc, 'b')
plt.ylabel('Accuracy ')
plt.xlabel('Number of neighbors (K)')
plt.show()

print("The best accuracy was with", mean_acc.max(initial=5), "with k=", mean_acc.argmax() + 1)
