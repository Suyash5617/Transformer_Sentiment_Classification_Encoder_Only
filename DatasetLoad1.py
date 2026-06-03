from sklearn.datasets import load_iris

def main():
    dataset = load_iris()

    print("Independent(feature) varible name are : ")
    print(dataset.feature_names)

    # print("Dependent (Lable)varible name are : ")
    # print(dataset.target_name)
    


if __name__=="__main__":
    main()    