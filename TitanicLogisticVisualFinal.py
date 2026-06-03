import pandas as pd
import numpy as np
import matplotlib as plt

from matplotlib.pyplot import figure, show
import seaborn as sns
from seaborn import countplot

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

def MartvellousTitanicLogistic(Datapath):
    df  = pd.read_csv(Datapath)

    print("Dataset successufully")
    print(df.head())

    print("Dimention of dataset is : ",df.shape)

    df.drop(columns=['Passengerid','zero'], inplace=True)

    print("Dimention of dataset is : ",df.shape)

    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

    figure()
    target = "Survived"
    countplot(data = df, x = target).set_title("Survived vs Non Survived")
    # show()

    figure()
    target = "Survived"
    countplot(data = df, x = target, hue='Sex').set_title("Based on gender")
    # show()

    figure()
    target = "Survived"
    countplot(data = df, x = target, hue='Pclass').set_title("Based on Pclass")
    # show()

    figure()
    df['Age'].plot.hist().set_title("Age report")
    # show()

    figure()
    df['Fare'].plot.hist().set_title("Fare report")
    # show()

    plt.figure(figsize = (10,6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title("Featur correlatiion Heatmap")
    plt.show()

    x =df.drop(columns = ['Survived'])
    y = df['Survived']

    print("Dimentions of Target : ",x.shape)
    print("Dimentions of Labels : ",x.shape)

    scaler = StandardScaler()
    x_scale = scaler.fit_transfrom(x)

    x_train, x_test, y_train, y_test = train_test_split(x_scale, y, test_size= 0.2, random_state=42)

    model = LogisticRegression()
    model.fit(x_train,y_train)

    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("Accuracy is : ",accuracy)
    print("Confusion matrix : ")
    print(cm)


def main():
    MartvellousTitanicLogistic("MarvellousTitanicDataset.csv")

if __name__ =="__main__":
    main()
    