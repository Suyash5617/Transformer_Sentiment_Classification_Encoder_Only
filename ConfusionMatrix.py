from sklearn.metrics import confusion_matrix

def main():
    Actual = [1,0,1,1,0,1,0,1,1,0]
    Predicted = [1,0,1,0,0,1,1,1,1,1]

    Con_Mat = confusion_matrix(Actual,Predicted)

    print("Confusion matrix is :")
    print(Con_Mat)


if __name__ == "__main__":
        main()