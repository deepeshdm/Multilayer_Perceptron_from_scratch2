

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from Neural_Network import neuralNetwork

df = pd.read_csv("Iris_dataset.csv")

y_dummies = pd.get_dummies(df.species)
df = pd.concat([df, y_dummies], axis=1)
df.drop("species", axis=1)

x = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]].values
y = df[["setosa", "versicolor", "virginica"]].values + 0.01

x_train, x_test, y_train, y_test = \
    train_test_split(x, y, test_size=0.2, random_state=0)


# -------------------------------------------------------------------------


def predict_label(x, model, get_species=False):
    output = model.predict(x)

    if get_species == True:
        if np.argmax(output) == 0:
            return "setosa"
        elif np.argmax(output) == 1:
            return "versicolor"
        else:
            return "virginica"
    else:
        return output


# -------------------------------------------------------------------------

# Calculating Accuracy


def get_accuracy(x_test, y_test, model):
    accurately_predicted = 0

    for i in range(len(x_test)):

        y_predicted = model.predict(x_test[i])
        actual_value = y_test[i]

        if np.argmax(y_predicted) == np.argmax(actual_value):
            accurately_predicted += 1
        else:
            pass

    accuracy = accurately_predicted / len(x_test)
    return accuracy


# ----------------------------------------------------------------------------

# Training Neural Network

nn = neuralNetwork(4, 4, 3, 0.5)
epochs = 5

for epoch in range(epochs):

    for i in range(len(x_train)):
        nn.train(x_train[i], y_train[i])
    pass

# ----------------------------------------------------------------------------

accuracy = get_accuracy(x_test,y_test,nn)
predicted_value = predict_label(x=[6.4,2.8,5.6,2.2],model=nn,get_species=True )

print("Accuracy : {}".format(accuracy))
print(predicted_value)


