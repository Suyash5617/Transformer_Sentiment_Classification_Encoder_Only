import numpy as np 
import pandas as pd 
from copy import deepcopy 
from matplotlib import pyplot as plt

def MarvellousKMean():
# Set three centers, the model should predict similar results
    center_1 = np.array([1,1])
    print(center_1)
    print("___________________________________")

    center_2 = np.array([5,5])
    print(center_2)
    print("___________________________________")

    center_3= np.array([8,1])
    print(center_3)
    print("___________________________________")

    #Generate random data and center it to the three centers
    data_1= np.random.randn(7, 2) + center_1
    print("Elements of first cluster with size"+str(len(data_1)))
    print(data_1)
    print("___________________________________")

    data_2 = np.random.randn(7,2) + center_2
    print("Elements of second cluster with size"+str(len(data_2)))
    print(data_2)
    print("___________________________________")

    data_3= np.random.randn(7,2) + center_3
    print("Elements of third cluster with size"+str(len(data_3)))
    print(data_3)
    print("___________________________________")

    data = np.concatenate((data_1, data_2, data_3), axis = 0)
    print("Size of complete data set"+str(len(data)))
    print(data)
    print("___________________________________")

    plt.scatter(data[:,0], data[:,1], s=7)
    plt.title('Marvellous Infosystems: Input Dataset')
    plt.show()
    #Number of clusters
    print("___________________________________")
    k=3

    # Number of training data
    n = data.shape[0]
    print("Total number of elements are",n) 
    print("___________________________________")

    #Number of features in the data 
    c = data.shape[1] 
    print("Total number of features are",c)
    print("___________________________________")

    # Generate random centers, here we use sigma and mean represent the whole data
    mean = np.mean(data, axis = 0)
    print("Value of mean", mean)
    print("___________________________________")

    # Calculate standard deviation
    std = np.std(data, axis = 0)
    print("Value of std", std)
    print("___________________________________")
    
    centers = np.random.randn(k,c)*std + mean
    print("Random points are", centers)
    print("___________________________________")

    # Plot the data and the centers generated as random
    plt.scatter(data[:,0], data[: ,1],c ='r' ,s=7)
    plt.scatter(centers[:,0],centers[:,1], marker ='*' , c='g' ,s = 150 )
    plt.title('Marvellous Infosystems: Input Datase with random centroid')
    plt.show()
    print("___________________________________")

    centers_old = np.zeros(centers.shape) # to store old centers
    centers_new = deepcopy(centers)

    #Store new centers
    print("Values of old centroids")
    print(centers_old)
    print("___________________________________")

    print("Values of new centroids") 
    print(centers_new)
    print("___________________________________")

    data.shape
    clusters =  np.zeros(n)
    distances = np.zeros((n,k))

    print("Initial distances are")
    print(distances)
    print("___________________________________")

    error = np.linalg.norm(centers_new - centers_old)
    print("value of error is",error)

    # When, after an update , the estement of that center stays the same, exit loop
    while error != 0:
        print("value of error is ", error)
        #Measure the distance to every center

        print("Measure the distance to every center")
        for i in range(k):
            print("Iteration number ",i)
            distances[:,i] = np.linalg.norm(data - centers[i], axis=1)
            
        #Assign all training data to closest center 
        clusters = np.argmin (distances, axis = 1)

        centers_old = deepcopy(centers_new)

        #Calculate mean for every cluster and update the center 
        for i in range(k):
            centers_new[i] = np.mean(data[clusters == 1], axis=0)
        error = np.linalg.norm(centers_new-centers_old)
    #end of while
    centers_new

    #Plot the data and the centers generated as random 
    plt.scatter(data[:,0],data[:,1], s=7)
    plt.scatter(centers_new[:,0],centers_new[:,1],marker = '*', c='g', s=150)
    plt.title('Marvellous Infosystems: Final data with Centroid')
    plt.show()
    
def main():
    print(" Marvellous Infosystems by Piyush Khairnar")

    print("Unsuervised Machine Learning")

    print("Clustering using K Mean Algorithm")

    MarvellousKMean()

if __name__=="__main__":
    main()


 