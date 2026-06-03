import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main():
    Dataset = pd.read_csv("iris.csv")
    X = Dataset.iloc[:,[0,1,2,3]].values
    
    WCSS = []
    for k in range (1,11):
        model = y_kmeans(n_clusters = k,init = 'k-means++',n_init = 10,random_state  = 42)
        model.fit(X)
        print(model.inertia_)#WCSS
        
        WCSS.append(model.inertia_)
    plt.plot(range(1,11),WCSS,marker = 'o')
    plt.title("Elbow method of KMEANS")
    plt.xlabel("Value of K")
    plt.ylabel("WCSS")
    plt.grid(True)
    plt.show()
    
    model = y_kmeans(n_clusters = 3,init = 'k-means++',n_init = 10,random_state  = 42)
    model.fit(X)
    y_kmeans = model.fit_predict(X)
    print("Values of y_kmeans")
    print(y_kmeans)
    
    plt.scatter(X[y_kmeans == 0,0],[y_kmeans == 0,1], s =100 ,c = 'red',label = 'Setosa')
    
    plt.scatter(X[y_kmeans == 1,0],[y_kmeans == 1,1], s =100 ,c = 'blue',label = 'Versicolor')
    
    plt.scatter(X[y_kmeans == 2,0],[y_kmeans == 2,1], s =100 ,c = 'green',label = 'Virginica')
    
    plt.scatter(model.cluster_centers_[:,0],model.cluster_centers_[:1], s= 100,c = 'yellow',label = 'centriod')
    plt.legend()
    plt.show()

if __name__ =="__main__":
    main()