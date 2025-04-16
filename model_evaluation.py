import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from ipywidgets import interact, interactive, fixed, interact_manual
from sklearn.model_selection import (train_test_split, cross_val_score,
                                     cross_val_predict)
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge

def DistributionPlot(RedFunction, BlueFunction, RedName, BlueName, Title):
    width = 12
    height = 10
    plt.figure(figsize=(width, height))
    
    ax1 = sns.kdeplot(RedFunction, color="r", label=RedName)
    ax2 = sns.kdeplot(BlueFunction, color="b", label=BlueName, ax=ax1)

    plt.title(Title)
    plt.xlabel('Price (in dollars)')
    plt.ylabel('Proportion of Cars')
    plt.show()
    plt.close()

def PollyPlot(xtrain, xtest, y_train, y_test, lr,poly_transform):
    width = 12
    height = 10
    plt.figure(figsize=(width, height))
    
    
    #training data 
    #testing data 
    # lr:  linear regression object 
    #poly_transform:  polynomial transformation object 
 
    xmax=max([xtrain.values.max(), xtest.values.max()])

    xmin=min([xtrain.values.min(), xtest.values.min()])

    x=np.arange(xmin, xmax, 0.1)


    plt.plot(xtrain, y_train, 'ro', label='Training Data')
    plt.plot(xtest, y_test, 'go', label='Test Data')
    plt.plot(x, lr.predict(poly_transform.fit_transform(x.reshape(-1, 1))), label='Predicted Function')
    plt.ylim([-10000, 60000])
    plt.ylabel('Price')
    plt.legend()

def split_training_and_testing_data():
    y_data = df['price']
    x_data=df.drop('price',axis=1) # all other numerical data from df

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.10, random_state=1)


    print("number of test samples :", x_test.shape[0])
    print("number of training samples:",x_train.shape[0])
    return (x_train, x_test, y_train, y_test)

def f(order, test_data):
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=test_data, random_state=0)
    pr = PolynomialFeatures(degree=order)
    x_train_pr = pr.fit_transform(x_train[['horsepower']])
    x_test_pr = pr.fit_transform(x_test[['horsepower']])
    poly = LinearRegression()
    poly.fit(x_train_pr,y_train)
    PollyPlot(x_train['horsepower'], x_test['horsepower'], y_train, y_test, poly,pr)

if __name__=='__main__':
    lre = LinearRegression()
    x_train, x_test, y_train, y_test = split_training_and_testing_data()
    lre.fit(x_train[['horsepower']], y_train) # always fit the model with train data
    lre.score(x_test[['horsepower']], y_test) # R^2 value 0.3635
    lre.score(x_train[['horsepower']], y_train) # R^2 value 0.6619

    # cross validation score
    y_data = df['price']
    x_data=df.drop('price',axis=1) # all other numerical data from df
    Rcross = cross_val_score(lre, x_data[['horsepower']], y_data, cv=4) # array([0.7746232 , 0.51716687, 0.74785353, 0.04839605])
    print("The mean of the folds are", Rcross.mean(), "and the standard deviation is" , Rcross.std())
    -1 * cross_val_score(lre,x_data[['horsepower']], y_data,cv=4,scoring='neg_mean_squared_error')

    yhat = cross_val_predict(lre,x_data[['horsepower']], y_data,cv=4)
    yhat[0:5]

    # Using distribution plot we can find out the difference between predicted car values and actual car values with 2 sets of 
    # data: train data, test data. For this we need multiple linear regression
    lr = LinearRegression()
    lr.fit(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_train)

    yhat_train = lr.predict(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])
    print(yhat_train[0:5])
    yhat_test = lr.predict(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])
    print(yhat_test[0:5])

    Title = 'Distribution  Plot of  Predicted Value Using Training Data vs Training Data Distribution'
    DistributionPlot(y_train, yhat_train, "Actual Car Price (Train)", "Predicted Car Price (Train)", Title)

    Title='Distribution  Plot of  Predicted Value Using Test Data vs Data Distribution of Test Data'
    DistributionPlot(y_test,yhat_test,"Actual Car Price (Test)","Predicted Car Price (Test)",Title)
    # so in the test data plot we found a big difference beteen the Actual and Predicted value
    # So we are going for the polynomial transformation
    # Overfitting occurs when the model fits the noise, but not the underlying process. 
    # Therefore, when testing your model using the test set, your model does not perform as well since it is modelling noise, 
    # not the underlying process that generated the relationship. Let's create a degree 5 polynomial model.
    pr = PolynomialFeatures(degree=5)
    x_train_pr = pr.fit_transform(x_train[['horsepower']])
    x_test_pr = pr.fit_transform(x_test[['horsepower']])
    
    poly = LinearRegression()
    poly.fit(x_train_pr, y_train)
    yhat = poly.predict(x_test_pr) # predicted value using test data for the poly model
    print(yhat[0:5])
    PollyPlot(x_train['horsepower'], x_test['horsepower'], y_train, y_test, poly,pr)
    poly.score(x_train_pr, y_train) # 0.55
    poly.score(x_test_pr, y_test) # -29.87

    # Let's see how the R^2 changes on the test data for different order polynomials and then plot the results:
    Rsqu_test = []

    order = [1, 2, 3, 4]
    for n in order:
        pr = PolynomialFeatures(degree=n)
        
        x_train_pr = pr.fit_transform(x_train[['horsepower']])
        
        x_test_pr = pr.fit_transform(x_test[['horsepower']])    
        
        lr.fit(x_train_pr, y_train)
        
        Rsqu_test.append(lr.score(x_test_pr, y_test))

    plt.plot(order, Rsqu_test)
    plt.xlabel('order')
    plt.ylabel('R^2')
    plt.title('R^2 Using Test Data')
    plt.text(3, 0.75, 'Maximum R^2 ') 


    # Ridge Regression
    # a degree two polynomial transformation on our data.
    pr=PolynomialFeatures(degree=2)
    x_train_pr=pr.fit_transform(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg','normalized-losses','symboling']])
    x_test_pr=pr.fit_transform(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg','normalized-losses','symboling']])
    RigeModel=Ridge(alpha=1)
    RigeModel.fit(x_train_pr, y_train)
    yhat = RigeModel.predict(x_test_pr)
    print('predicted:', yhat[0:4])
    print('test set :', y_test[0:4].values)

    #Create a Ridge Regression model and evaluate it using values of the hyperparameter alpha ranging from 0.001 
    # to 1 with increments of 0.001. Create a list of all Ridge Regression R^2 scores for training and testing data.
    Rsqu_test = []
    Rsqu_train = []
    Alpha = np.arange(0.001,1,0.001)
    pbar = tqdm(Alpha)

    for alpha in pbar:
        RigeModel = Ridge(alpha=alpha) 
        RigeModel.fit(x_train_pr, y_train)
        test_score, train_score = RigeModel.score(x_test_pr, y_test), RigeModel.score(x_train_pr, y_train)
        pbar.set_postfix({"Test Score": test_score, "Train Score": train_score})
        Rsqu_test.append(test_score)
        Rsqu_train.append(train_score)

    from sklearn.model_selection import GridSearchCV
    from sklearn.linear_model import Ridge
    parameters= [{'alpha': [0.001,0.1,1, 10, 100, 1000, 10000, ...]}]
    RR=Ridge()
    Grid1 = GridSearchCV(RR, parameters,cv=4) 
    Grid1.fit(x_data[['attribute_1', 'attribute_2', ...]], y_data)
    BestRR=Grid1.best_estimator_
    BestRR.score(x_test[['attribute_1', 'attribute_2', ...]], y_test)