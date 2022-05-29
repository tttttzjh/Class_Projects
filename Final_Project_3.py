""" Tyler Zheng
    ITP-449
    Final Project - 3
    Description: this program builds a classficaton model that predicts the edibility of mushrooms.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay


def mushroom():
    # pd.set_option('display.width', None)
    # file_path = 'mushrooms.csv'
    # df_mushrooms = pd.read_csv(file_path)
    # # print(df_mushrooms.head())
    #
    # y = df_mushrooms['class']
    # x = df_mushrooms.drop(['class'], axis=1)
    # X = pd.get_dummies(x)
    #
    # X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42, stratify=y)
    #
    # model_dt = DecisionTreeClassifier(criterion='entropy', random_state=42, max_depth=6)
    # model_dt.fit(X_train, y_train)
    # print('1. The classification tree is being built\n')
    # print(
    #     '2. The dataset is partitioned:\n\ta. random_state=42\n\tb. Partitions 70/30\n\tc. Stratify y\n\td. max_depth=6\n\te. Use entropy\n')
    #
    # y_pred = model_dt.predict(X_test)
    #
    # print('4. The confusion matrix is:\n')
    # conf_matrix = confusion_matrix(y_test, y_pred)
    # print(conf_matrix, '\n')
    #
    # print('   The plot of confusion matrix is shown in mushroom_cm.png\n')
    # cm_disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=model_dt.classes_)
    # fig, ax1 = plt.subplots()
    # cm_disp.plot(ax=ax1)
    # plt.tight_layout()
    # plt.savefig('mushroom_cm.png')
    #
    # # class_rep = classification_report(y_test, y_pred)
    # # print('classification report:\n', class_rep)
    #
    # score_test = model_dt.score(X_test, y_test)
    # score_train = model_dt.score(X_train, y_train)
    # print('5. The accuracy score on the training partition is', score_train, '\n')
    # print('6. The accuracy score on the test partition is', score_test, '\n')
    #
    # # plot_tree(model_dt)
    # plot_tree(model_dt, feature_names=X.columns, class_names=y.unique(), filled=True)
    # plt.tight_layout()
    # plt.savefig('Mushroom_Tree.png')
    # print('The classification tree is shown in the Mushroom_Tree.png\n')
    #
    # # Top three most important features
    # fi = model_dt.feature_importances_
    # features = X.columns
    #
    # fi_sorted = fi.tolist()
    # fi_sorted.sort(reverse=True)
    #
    # top_scores = []
    # top_features = []
    # for i in range(3):
    #     top_scores.append(fi_sorted[i])
    #
    # for j in range(len(top_scores)):
    #     for k in range(len(fi)):
    #         if top_scores[j] == fi[k]:
    #             top_features.append(features[k])
    # print('8. The top three most important features:\n', top_features)
    # print('   The top three most important feature scores:\n', top_scores)

    new_mushroom = {'cap-shape': 'x', 'cap-surface': 's', 'cap-color': 'n', 'bruises': 't', 'odor': 'y',
                    'gill-attachment': 'f', 'gill-spacing': 'c', 'gill-size': 'n', 'gill-color': 'k',
                    'stalk-shape': 'e',
                    'stalk-root': 'e', 'stalk-surface-above-ring': 's', 'stalk-surface-below-ring': 's',
                    'stalk-color-above-ring': 'w',
                    'stalk-color-below-ring': 'w', 'veil-type': 'p', 'veil-color': 'w', 'ring-number': 'o',
                    'ring-type': 'p',
                    'spore-print-color': 'r', 'population': 's', 'habitat': 'u'}

    new_mushroom_pd = pd.DataFrame(new_mushroom)


if __name__ == '__main__':
    mushroom()