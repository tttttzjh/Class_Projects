""" Tyler Zheng
    ITP-449
    Assignment 10
    Description: In this program, the wine quality data set will be analyzed and the wines
    will be grouped into several clusters.
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def main():
    # import the data table
    pd.set_option('display.width', None)
    file_path = 'wineQualityReds.csv'
    df_wines = pd.read_csv(file_path)

    # make the x and y table
    df_wines = df_wines.drop(['Wine'], axis=1)
    df_wines_quality = df_wines['quality']
    df_wines = df_wines.drop(['quality'], axis=1)

    # normalize all the columns
    norm = StandardScaler()
    X = pd.DataFrame(norm.fit_transform(df_wines), columns=df_wines.columns)

    # Iterate on the k values and store the inertia_ for each clustering
    ks = range(1, 11)
    inertia = []
    for k in ks:
        model = KMeans(n_clusters=k)
        model.fit(X)
        inertia.append(model.inertia_)

    # plot the chart of inertia vs number of clusters k
    plt.plot(ks, inertia, '-o')
    plt.xlabel('Number of Cluster (k)')
    plt.ylabel('Inertia')
    plt.title('Inertia vs Number of Clusters')

    plt.savefig('Inertia vs Number of Clusters.png')

    # based on the graph, choose K=6
    model_new = KMeans(n_clusters=6, random_state=2021)
    model_new.fit(X)
    labels = model_new.labels_
    # add the cluster number as well as the quality back to the dataframe
    df_wines['Cluster number'] = pd.Series(labels)
    df_wines['quality'] = df_wines_quality
    # print a crosstab of cluster number vs quality
    print(pd.crosstab(df_wines['Cluster number'], df_wines['quality']))

    # From the crosstab result, we can see that the clusters do not represent the quality of wine
    # very well because for each cluster number, the number of occurances of that number is relatively
    # spread out in each wine quality number. If the clusters are good representation, the occurances should
    # be in a certain wine quality number.


if __name__ == '__main__':
    main()