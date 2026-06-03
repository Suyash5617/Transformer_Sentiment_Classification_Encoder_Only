import pandas as pd
import numpy as np
from sklearn.cluster import k_means
from matplotlib.pyplot import plt

def main():
    dataset = pd .read_csv('iris.csv')
    X = dataset.iloc[:,[0,1,2,3]].values

    
     

if  __name__=="__main__":
    main()
