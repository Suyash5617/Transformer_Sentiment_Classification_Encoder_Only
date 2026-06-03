import pandas as pd 
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


def MarvellousBrainLinear(Datapath):
    Line = "*"*50

    df = pd.read_csv(Datapath)
    
    print(Line)
    print("First fev record of the dataset are : ")
    print(Line)
    print(df.head())
    print(Line)

    print("Statistical information of the dataset : ")
    print(Line)
    print(df.describe())
    print(Line)

    x = df[['Head Size(cm^3)']]
    y = df[['Brain Weight(grams)']]

    print("Independent variables are : Head Size")
    print("Dependent variables are : Brain Weigth")

    print("Total records in dataset :  ",x.shape)

    x_train, x_test, y_train, y_test = train_test_split(x,y ,test_size= 0.2, random_state=42)
    print("Dimention of Training dataset : ",x_train.shape)
    print("Dimention of Training dataset : ",x_test.shape)

    model = LinearRegression()
    model.fit(x_train,y_train)

    y_pred = model.predict(x_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test,y_pred)

    print("Visual representation")

    plt.figure(figsize= (8,5))
    plt.scatter(x_test,y_test, color = 'blue', label= 'Actual')
    plt.plot(x_test.values.flatten(), y_pred, color='red', linewidth = 2, label='RegrassionLine')
    plt.xlabel('Head Size(cm^3)')
    plt.ylabel('Brain Weight(grams)')
    print("Marvellous Head Brain Regression")
    plt.legend()
    plt.grid(True)
    plt.show()

    print("Result of case study")
    print("Slope of line  (m) : ", model.coef_[0])
    print("Intercept (c) : ",model.intercept_)
    print("Mean Squared Error is : ", mse)
    print("Root Mean Squared Error is  : ",rmse)
    print("R Squread Value : ", r2)

def main():
    MarvellousBrainLinear("MarvellousHeadBrain.csv")

if __name__ =="__main__":
    main()    