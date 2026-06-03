from sklearn.datasets import load_iris

def main():
    dataset = load_iris()

    print("Element from the dataset are : ")

    for i in range(len(dataset.target)):
        print("ID %d, Feature %s, Lable : %s" %(i, dataset.data[i], dataset.target[i]))

      
if __name__=="__main__":
    main()    