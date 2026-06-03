import pandas as pd

def main():
    url = "http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data" 

    names = ["sepal_length", "sepal_Width","petal_length","petal_width","class"]

    dataset = pd.read_csv(url,names = names)
    print(dataset.head())

if __name__ =="__main__":
    main()

