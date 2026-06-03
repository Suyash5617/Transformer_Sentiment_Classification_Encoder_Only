from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main():
    Dataset = pd.read_csv("iris.csv")
    X = Dataset.iloc[:,[0,1,2,3]].values
    print(X.shape)

    model = KMeans(n_clusters = 3,init = 'k-means++',n_init = 10,random_state  = 42)
    model.fit(X)
    y_kmeans = model.fit_predict(X)
    
   
    print(y_kmeans.shape)

    
    print("Cluster of setosa : 1")
    for i in range(0,10):
        print(X[i], y_kmeans[i])

    print("Cluster of versicolor : 1")
    for i in range(51, 61):
        print(X[i], y_kmeans[i])    

    print("Cluster of verginica: 1")
    for i in range(101, 111):
        print(X[i], y_kmeans[i])     


if __name__ =="__main__":
    main()