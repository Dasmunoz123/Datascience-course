import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

##
## Loading dataset in dataframe
fruits = pd.read_table('../data/fruit_data_with_colors.txt')
X_fruits = fruits[['height', 'width', 'mass', 'color_score']]
y_fruits = fruits['fruit_label']
fruits_name = dict(zip(fruits['fruit_label'].unique(), fruits['fruit_name'].unique()))   

##
## Exploring dataset characteristics
## Fig scatter matrix
from matplotlib import cm
cmap = cm.get_cmap('gnuplot')
figure_1 = pd.plotting.scatter_matrix(X_fruits, c= y_fruits, marker = 'o', s=40, hist_kwds={'bins':20}, figsize=(9,9), cmap=cmap)
plt.show()

## Fig 3d scatter
from mpl_toolkits.mplot3d import Axes3D
figure_2 = plt.figure()
ax = figure_2.add_subplot(111, projection = '3d')
ax.scatter(X_fruits['width'], X_fruits['height'], X_fruits['color_score'], c = y_fruits, marker = 'o', s=100)
ax.set_xlabel('width')
ax.set_ylabel('height')
ax.set_zlabel('color_score')
plt.show()


##
## Training knn
X_train, X_test, y_train, y_test = train_test_split(X_fruits, y_fruits, random_state=0)
scaler = MinMaxScaler() # we must apply the scaling to the test set that we computed for the training set
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_train_scaled, y_train)
print('Accuracy of K-NN classifier on training set: {:.2f}'.format(knn.score(X_train_scaled, y_train)))
print('Accuracy of K-NN classifier on test set: {:.2f}'.format(knn.score(X_test_scaled, y_test)))


##
## Testing with new unseen data
example_fruit = [[5.5, 2.2, 10, 0.70]]
example_fruit_scaled = scaler.transform(example_fruit)
print('Predicted fruit type for ', example_fruit, ' is ', fruits_name[knn.predict(example_fruit_scaled)[0]])