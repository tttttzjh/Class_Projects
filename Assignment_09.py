""" Tyler Zheng
    ITP-449
    Assignment 9
    Description: In this project, we will analyze the RMS titanic dataset and
    classify survivability based on the various factors of the passengers
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report


def main():
    # import the data, save as a dataframe
    file_path = "titanicTrain.csv"
    pd.set_option('display.width', None)
    df_titanic = pd.read_csv(file_path)

    # only select the needed columns
    df_titanic = df_titanic[['Survived', 'Pclass', 'Sex', 'Age']]
    # drop all the rows with missing values
    df_titanic = df_titanic.dropna()

    # set the x and y
    y = df_titanic['Survived']
    x = df_titanic[['Pclass', 'Sex', 'Age']]

    # first plot, histograms of every variable
    fig, ax = plt.subplots(2, 2)
    ax[0, 0].hist(y)
    ax[0, 0].set(xlabel='Survived', ylabel='Count')
    ax[0, 1].hist(x['Pclass'])
    ax[0, 1].set(xlabel='Pclass', ylabel='Count')
    ax[1, 0].hist(x['Sex'])
    ax[1, 0].set(xlabel='Sex', ylabel='Count')
    ax[1, 1].hist(x['Age'])
    ax[1, 1].set(xlabel='Age', ylabel='Count')
    fig.suptitle('Titanic Data: Histograms of Input Variables')
    fig.tight_layout()

    plt.savefig('Titanic.png')

    # then convert categorical values into dummy variables
    X = pd.get_dummies(x)

    # separate the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.70, random_state=76)

    # initialize the model
    model_logreg = LogisticRegression(max_iter=1000)

    # train the model
    model_logreg.fit(X_train, y_train)

    # make predictions
    y_pred = model_logreg.predict(X_test)

    # calculate accuracy score
    accuracy_score = model_logreg.score(X_test, y_test)

    # print confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=model_logreg.classes_)
    cm_disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model_logreg.classes_)
    # plot the confusion matrix in the second plot
    fig, ax1 = plt.subplots()
    cm_disp.plot(ax=ax1)
    # add title to the plot
    ax1.set(title='Titanic Survivability\n(Model Accuracy:' + str(round(accuracy_score * 100, 2)) + '%)')
    plt.savefig('Titanic_cm.png')

    # prediction for a 30 year old male passenger in 3rd class
    # create the input dataframe
    prediction_input = {'Pclass': [3], 'Age': [30.0], 'Sex_female': [0], 'Sex_male': [0]}
    df_prediction = pd.DataFrame(prediction_input)
    # make predictions
    survive_pred = model_logreg.predict(df_prediction)
    # print out the message
    print('Prediction for 30-year-old male passenger in 3rd class:', survive_pred[0])


if __name__ == '__main__':
    main()