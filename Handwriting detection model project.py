## This program is a model that classifies the handwritten digits from 1 to 9
## Code by Mahmoud-K-Ismail as a part of Artificial intellegence course

from sklearn.datasets import load_digits
import sklearn
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R
from sklearn.decomposition import PCA
from matplotlib import cm
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

#loading the handwritten digits
digits = load_digits()
X = digits.data
print("Shape of the data X:", X.shape)
#reshaping each data element to be 8x8 matrix
print("First image values reshaped to 8 x 8\n", X[0].reshape(8, 8))
# obtaining the labels 
y = digits.target
print("Shape of the labels:", y.shape)
#print("The first 10 labels:", y[:10])

#printing the actual photos instead of numbers to visualize the problem and their corresponding labels

f, axes = plt.subplots(2, 5, figsize=(15, 5))  # 2 rows, 5 columns
for ii, ax in enumerate(axes.flatten()):
    ax.imshow(X[ii].reshape(8, 8), cmap="Greys")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"Truth: {y[ii]}")
plt.show()

#reducing the diminsions of data to 2 only, for easy computations
pca2 = PCA(2)
pca2.fit(X)
dt_PCA = pca2.transform(X)
X_pca = dt_PCA
# shows the graph of the data after transforming it to two diminsions only
print(dt_PCA.shape)
plt.plot(dt_PCA)

## plot the data and its class
colors = cm.Paired(np.linspace(0., 1., 10)) # list of 10 colors

colors_all = colors[digits.target]

plt.figure()
plt.scatter(X_pca[:, 0], X_pca[:, 1], color=colors_all)
plt.show()

# splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, digits.target, test_size=0.5, shuffle=False
)

lda = LDA()
lda.fit(X_train, y_train)

y_predict = lda.predict(X_test)

#printing the predicted values
#print(y_predict)

#function that calculates the accuracy given the two outputs and the percentage
def acc(real,predicted,x):
    n = 0
    for i in range (x):
        if (real[i] == predicted[i]):
            n+=1
    result = (n/x)*100
    return result
print (f"The accuracy is : {acc(y_test,y_predict,len(y_test))}")

# A function that summerizes all of the previous steps
# It takes the fraction of testing sample and outputs list of the accuracies
def accuracy(test_size):
    digits = load_digits()
    X = digits.data
    pca2 = PCA(2)
    pca2.fit(X)
    dt_PCA = pca2.transform(X)
    X_pca = dt_PCA
    X_train, X_test, y_train, y_test = train_test_split(
    X, digits.target, test_size=0.5, shuffle=False
    )

    lda = LDA()
    lda.fit(X_train, y_train)

    y_predict = lda.predict(X_test)
    x = test_size*len(y_test)
    accuracy = acc(y_test,y_predict,int (x))
    return accuracy


test_sizes = np.linspace(0.1, 0.9, 10)

print (test_sizes)

accuracies = []

for test_size in test_sizes:
    accuracies.append(accuracy(test_size))
print (list(accuracies))
    

plt.plot(test_sizes,accuracies)
plt.show()