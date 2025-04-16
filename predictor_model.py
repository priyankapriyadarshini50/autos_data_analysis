import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

class PredictorModel:
    '''
    will predict the price of the car using the variables or features
    '''
    def __init__(self, url=None):
        self.url = url
        self.dframe = pd.read_csv(url)
        self.lm = LinearRegression()
        # print(self.dframe.head())

    def simple_linear_regression(self):
        '''
        one independent var(x1) as input
        one target var(y) as the output/predicted value
        y = b0 + b1x1
        a refers to the intercept of the regression line, in other words: the value of Y when X is 0
        b refers to the slope of the regression line, in other words: the value with which Y changes when X increases by 1 unit
        '''

        x = self.dframe[['highway_L/100km']]
        y = self.dframe[['price']]
        self.lm.fit(x, y)
        print(f"intercept={self.lm.intercept_}, slope={self.lm.coef_}") #38423.30585815743, arrayS
        Yhat = self.lm.predict(x)
        print(f"Some predicted values={Yhat[:5]}")

        

    def multiple_linear_regression(self):
        '''
        one continuous/predicted value Y
        two or more independent variable X
        Y = b0+b1*x1 + b2*x2 + b3*x3 + b4*x4
        '''
        Z = self.dframe[['horsepower', 'curb_weight', 'engine_size', 'highway_L/100km']]
        y = self.dframe[['price']]
        self.lm.fit(Z, y)
        print(f"intercept={self.lm.intercept_}, slope={self.lm.coef_}") # lm.coef_ is an array
        Yhat = self.lm.predict(Z) # predicted values
        self.distribution_plot(Yhat)
        r_square = self.lm.score(Z, y)
        print(f"R square value={r_square}")

    def polynomial_regression(self):
        x = self.dframe['highway-mpg']
        y = self.dframe['price']
        # Here we use a polynomial of the 3rd order (cubic) 
        f = np.polyfit(x, y, 3) # returns array of polynomial coefficients [b0, b1, b2, b3]
        p = np.poly1d(f)
        print(p)
        self.PlotPolly(p, x, y, 'highway-mpg')

        r_squared = r2_score(y, p(x))
        print('The R-square value is: ', r_squared)

        mean_squared_error(self.dframe['price'], p(x))


    def distribution_plot(self, model):
        '''
        Visualize the MLR model with distribution plot
        '''
        
        # plt.figure(figsize=(width, height))

        ax1 = sns.kdeplot(self.dframe['price'], hist=False, color="r", label="Actual Value")
        sns.kdeplot(model, hist=False, color="b", label="Fitted Values" , ax=ax1)


        plt.title('Actual vs Fitted Values for Price')
        plt.xlabel('Price (in dollars)')
        plt.ylabel('Proportion of Cars')
        plt.legend(['Actual Value', 'Predicted Value'])

        plt.show()
        plt.close()

    def PlotPolly(model, independent_variable, dependent_variabble, Name):
        x_new = np.linspace(15, 55, 100)
        y_new = model(x_new)

        plt.plot(independent_variable, dependent_variabble, '.', x_new, y_new, '-')
        plt.title('Polynomial Fit with Matplotlib for Price ~ Length')
        ax = plt.gca()
        ax.set_facecolor((0.898, 0.898, 0.898))
        fig = plt.gcf()
        plt.xlabel(Name)
        plt.ylabel('Price of Cars')

        plt.show()
        plt.close()

    def polynomial_transfer(self):
        '''
        Typically used as a preprocessing step for linear models.
        Basically, to reduce the noise (the difference between actual value and predicted value)
        
        '''
        pr=PolynomialFeatures(degree=2)
        Z = self.dframe[['horsepower', 'curb_weight', 'engine_size', 'highway_L/100km']]
        Z_pr=pr.fit_transform(Z)
        Z.shape # provides the dimension 
        Z_pr.shape

    def pipeline(self):
        '''
        Data Pipelines simplify the steps of processing the data. 
        We use the module Pipeline to create a pipeline. We also use StandardScaler as a step in our pipeline.
        '''
        Input=[('scale',StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)), ('model',LinearRegression())]
        pipe=Pipeline(Input)
        print(pipe)
        Z = self.dframe[['horsepower', 'curb_weight', 'engine_size', 'highway_L/100km']]
        Z = Z.astype(float)
        pipe.fit(Z,self.dframe['price'])
        ypipe=pipe.predict(Z)
        ypipe[0:4]
    
    def find_R_squared(self, independent_variable, dependent_variabble):
        # find R^2
        self.lm.score(independent_variable, dependent_variabble)

    def find_mean_square_error(self, dependent_variabble, Yhat):
        mse = mean_squared_error(dependent_variabble, Yhat)
        print('The mean square error of price and predicted value is: ', mse)
