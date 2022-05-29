""" Tyler Zheng
    ITP-449
    Final Project - 4
    Description: This program will
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression


def vehicle():
    pd.set_option('display.width', None)
    file_path = 'auto-mpg.csv'
    df_vehicle = pd.read_csv(file_path)

    mean_mpg = df_vehicle['mpg'].mean()
    median_mpg = df_vehicle['mpg'].median()
    print('1. Mean value of mpg:\n', mean_mpg)
    print('2. Median value of mpg:\n', median_mpg)

    print(
        '3. We can see that the mean value is higher, meaning that the mpg value should have a skewness to the right (positively skewed)')
    print('plot to verify')
    plt.figure(0)
    plt.hist(df_vehicle['mpg'], bins=20)
    plt.savefig('mpg value.png')

    # 4. plot the pairplot matrix

    df_vehicle = df_vehicle.drop(['No'], axis=1)
    df_vehicle = df_vehicle.drop(['car_name'], axis=1)
    dim = len(df_vehicle.columns)
    column_name = df_vehicle.columns.tolist()
    fig, ax = plt.subplots(dim, dim, figsize=[32, 24])
    for i in range(dim):
        for j in range(dim):
            if i != j:
                ax[i, j].scatter(df_vehicle.iloc[:, i], df_vehicle.iloc[:, j])
                ax[i, j].set(xlabel=column_name[i], ylabel=column_name[j])
            else:
                ax[i, j].hist(df_vehicle.iloc[:, i], bins=10)
    plt.figure(1)
    plt.tight_layout()
    plt.savefig('Vehicle_pairplot.png')

    print('5. Based on the pairplot matrix, the most strongly linearly correlated attributes are \n')
    print('6. Based on the pairplot matrix, the most weakly linearly correlated attributes are \n')

    # 7. scatterplot of mpg vs. displacement
    plt.figure(2)
    plt.scatter(df_vehicle['displacement'], df_vehicle['mpg'])
    plt.xlabel('displacement')
    plt.ylabel('mpg')
    plt.savefig('mpg vs displacement.png')

    # 8. build lineear regression model
    x = df_vehicle['displacement']
    y = df_vehicle['mpg']

    model_linreg = LinearRegression()

    X = x.values.reshape(-1, 1)

    model_linreg.fit(X, y)

    y_pred = model_linreg.predict(X)
    residuals = y - y_pred

    intercept = model_linreg.intercept_
    coef = model_linreg.coef_
    print('8.(1) model intercept B0: ', intercept, '\n')
    print('8.(2) model coefficient B1: ', coef[0], '\n')
    print('8.(3) Regression equation as per the model: mpg = displacement *', coef[0], '+', intercept, '\n')
    print('8.(4) For our modeel, the predicted value for mpg decreases as the displacement increases.\n')

    # given displacement value of 220
    displacement = 220
    pred_mpg = displacement * coef[0] + intercept
    print('8.(5) The predicted mpg value for a car with a displacement value of 220 is', pred_mpg, '\n')

    # 8. (6)(7) scatter plot and residual plot
    fig, ax = plt.subplots(1, 2)
    ax[0].scatter(X, y)
    ax[0].plot(X, y_pred)
    ax[0].set(xlabel='displacement', ylabel='mpg')

    ax[1].scatter(y_pred, residuals)
    ax[1].set(xlabel='Predicted mpg', ylabel='residuals')

    plt.tight_layout()
    plt.savefig('mpg vs displacement analysis.png')


if __name__ == '__main__':
    vehicle()