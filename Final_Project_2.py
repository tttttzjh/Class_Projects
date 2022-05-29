""" Tyler Zheng
    ITP-449
    Final Project - 2
    Description: This program analyzes the universal bank data and makes predictions using decision trees.
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay


def loan():
    # import the dataset
    pd.set_option('display.width', None)
    file_path = 'UniversalBank.csv'
    df_bank = pd.read_csv(file_path)
    # print(df_bank.head())

    print('1. The target variable is Personal Loan\n')

    print('2. The attributes Row and Zip code are being removed\n')
    df_bank = df_bank.drop(['Row'], axis=1)
    df_bank = df_bank.drop(['ZIP Code'], axis=1)

    # create x and y
    y = df_bank['Personal Loan']
    X = df_bank.drop(['Personal Loan'], axis=1)

    # split the dataset
    print('3. The dataset is partitioned:\n\ta. Random_state=42\n\tb. Partitions 70/30\n\tc. Stratify y\n')
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42, stratify=y)

    # train the model, make predictions
    model_dt = DecisionTreeClassifier(criterion='entropy', random_state=42, max_depth=5)
    model_dt.fit(X_train, y_train)
    y_pred_test = model_dt.predict(X_test)
    y_pred_train = model_dt.predict(X_train)

    print('5. In order to see the number of cases in the training partition represented')
    print('   people who accepted offers of a personal loan, the classification report')
    print('   for the training partition is being printed first:\n')
    class_rep_train = classification_report(y_train, y_pred_train)
    class_rep_test = classification_report(y_test, y_pred_test)
    print('   classification report:\n', class_rep_train)
    print('   From the report, we can see that the number of cases is 336\n')

    # plot the decision tree
    print('6. The decision tree is being shown in the UniversalBank_Tree.png\n')
    y_unique = ['No', 'Yes']  # since it contains 1 and 0, representing no and yes
    plot_tree(model_dt, feature_names=X.columns, class_names=y_unique, filled=True)
    plt.tight_layout()
    plt.savefig('UniversalBank_Tree.png')

    print('7. In order to see the number of the misclassification, the confusion matrix is being printed:\n')

    conf_matrix_train = confusion_matrix(y_train, y_pred_train)
    print(conf_matrix_train, '\n')
    print('   From the confusion matrix for the training partition, the number of acceptors that the model')
    print('   classifies as non-acceptors is 30.\n')

    print('8. From the confusion matrix for the training partition, the number of non-acceptors that the')
    print('   model classifies as acceptors is 21.\n')

    # find the accuracy score
    score_test = model_dt.score(X_test, y_test)
    score_train = model_dt.score(X_train, y_train)
    print('9. The accuracy score on the training partition:', score_train, '\n')
    print('10. The accuracy score on the test partition:', score_test, '\n')


if __name__ == '__main__':
    loan()