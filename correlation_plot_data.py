import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot
from scipy import stats
from import_data import ImportCleaningDataSet

class DataCorrelation:

    def __init__(self, url=None):
        self.url = url
        self.dframe = pd.read_csv(url)
        # print(self.dframe.head())

    def scattered_plot(self):
        '''
        relationship between continues variables such as "engine-size" and "price".(contineous)
        Engine size as potential predictor variable of price
        '''
        width = 12
        height = 10
        pyplot.figure(figsize=(width, height)) # customize the fig size
        sns.regplot(x="engine_size", y="price", data=self.dframe)
        pyplot.ylim(0,)
        pyplot.show()
        print(self.dframe[["engine_size", "price"]].corr())

    def box_plot(self):
        '''
        relationship between categorical variables such as 
        "body-style" and "price" (body-style categorical variable)
        '''
        sns.boxplot(x="body-style", y="price", data=self.dframe)

    def data_value_count(self):
        '''
        provide correlation between cotegorial var like drive_wheels
        '''

        drive_wheels_counts = self.dframe['drive_wheels'].value_counts().to_frame()
        drive_wheels_counts.reset_index(inplace=True)
        drive_wheels_counts=drive_wheels_counts.rename(columns={'drive_wheels': 'value_counts'})
        
        # let's rename the index to 'drive-wheels':
        drive_wheels_counts.index.name = 'drive_wheels'
        print(drive_wheels_counts)

        # engine-location as variable
        engine_loc_counts = self.dframe['engine_location'].value_counts().to_frame()
        engine_loc_counts.reset_index(inplace=True)
        engine_loc_counts.rename(columns={'engine_location': 'value_counts'}, inplace=True)
        engine_loc_counts.index.name = 'engine_location'
        #print(engine_loc_counts.head(10))

    def data_group_by_pivot_plot(self):
        '''
        group by the variable "drive-wheels
        which type of drive wheel is most valuable, we can group "drive-wheels" and then average them
        calculate the average price for each of the different categories of data
        '''
        self.dframe['drive_wheels'].unique()

        df_group_one = self.dframe[['drive_wheels','body-style','price']]
        df_grouped = df_group_one.groupby(['drive_wheels'], as_index=False).agg({'price': 'mean'})
        print(df_grouped)

         # let's group by both 'drive-wheels' and 'body-style'
        # grouping results
        df_gptest = self.dframe[['drive_wheels','body-style','price']]
        grouped_test1 = df_gptest.groupby(['drive_wheels','body-style'],as_index=False).mean()
        print(grouped_test1)

        # This grouped data is much easier to visualize when it is made into a pivot table.
        # convert df to a pivot table
        grouped_pivot = grouped_test1.pivot(index='drive_wheels',columns='body-style')
        grouped_pivot = grouped_pivot.fillna(0) #fill missing values with 0
        print(grouped_pivot)

        # heat map based on the drive-wheel and body-style vs price
        fig, ax = pyplot.subplots()
        im = ax.pcolor(grouped_pivot, cmap='RdBu')

        #label names
        row_labels = grouped_pivot.columns.levels[1]
        col_labels = grouped_pivot.index

        #move ticks and labels to the center
        ax.set_xticks(np.arange(grouped_pivot.shape[1]) + 0.5, minor=False)
        ax.set_yticks(np.arange(grouped_pivot.shape[0]) + 0.5, minor=False)

        #insert labels
        ax.set_xticklabels(row_labels, minor=False)
        ax.set_yticklabels(col_labels, minor=False)

        #rotate label if too long
        pyplot.xticks(rotation=90)

        fig.colorbar(im)
        pyplot.show()

    def pearson_correlation(self):
        '''
        Let's calculate the Pearson Correlation Coefficient and P-value of 'wheel-base' and 'price'.
        '''
    
        pearson_coef, p_value = stats.pearsonr(self.dframe['wheel_base'], self.dframe['price'])
        print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)
    
    def model_visualization_using_residual_plot(self):

        width = 12
        height = 10
        pyplot.figure(figsize=(width, height))
        sns.residplot(x=self.dframe['highway_L/100km'], y=self.dframe['price'])
        pyplot.show()


if __name__=='__main__':
    dc = DataCorrelation(f"C:/Users/priya/OneDrive/Documents/Home_learning_django/ML_NumPy_stack/autos/automobile.csv")
    dc.scattered_plot()
    dc.data_group_by_pivot_plot()
    dc.pearson_correlation()
    dc.model_visualization_using_residual_plot()

# import piplite
# await piplite.install('seaborn')
# import matplotlib.pyplot as plt
# import seaborn as sns
# %matplotlib inline 
# plt.plot(x,y)

# scatter plot, there are multiple values of the dependent variable (y) for a unique value of an independent variable(x)
# single x multiple y (1,2), (1,7) (1,5) (1,9)
# plt.scatter(x,y)

# #Histogram, data in categorial form, binned form
# plt.hist(x,bins)

# #Bar plot
# plt.bar(x, height)

# # Pseudo Color Plot
# plt.pcolor(C)