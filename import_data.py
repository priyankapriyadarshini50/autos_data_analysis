import pickle
import pandas as pd
import numpy as np
from matplotlib import pyplot
import settings


class ImportCleaningDataSet:
    '''
    This class reads and cleaned the auto dataset and creates a pandas dataframe
    '''

    def __init__(self, dataframe=None):
        self.df = dataframe

    def reading_car_dataset(self):
        """
        Data cleaning Step1
        reading the dataset, statistical summary
        """

        headers = [
            "symboling", "normalized_losses", "make", "fuel_type", "aspiration", "num_of_doors",
            "body-style", "drive_wheels", "engine_location", "wheel_base", "length", "width",
            "height", "curb_weight", "engine_type", "num_of_cylinders", "engine_size",
            "fuel_system", "bore", "stroke", "compression-ratio", "horsepower", "peak_rpm",
            "city_mpg", "highway_mpg", "price" 
        ]

        self.df = pd.read_csv(settings.CAR_DATASET_URL, header=None)
        self.df.columns = headers
        print(self.df.head())

    def save_the_modified_dataframe(self):
        '''
        save the modified data set to the local machine
        preserve/save the modified data set
        '''
        self.df.to_csv(settings.LOCAL_FILE_PATH, index=False)

    def pickle_dataframe(self):
        '''
        pickling the data frame
        '''
        try:
            with open('automobile.pkl', 'wb') as file:
                pickle.dump(self.df, file)
        except pickle.PicklingError as e:
            print(f"Pickling Errror={e}")

    def identify_handle_missing_data(self):
        """
        Data cleaning Step2
        replace missing value with the newly calculated values
        Convert '?' to nan
        dataframe.replace(missing_value, new_value)

        """
        self.df.replace("?", np.nan, inplace=True)

    def evaluate_and_deal_missing_data(self):
        '''
        Data cleaning Step3
        Finding the missing data in dataframe (TRUE value for missing data)
        Count missing values in each column
        "normalized_losses": 41 missing data -> replace by mean
        "num-of-doors": 2 missing data -> replace by four
        "bore": 4 missing data -> replace by mean
        "stroke" : 4 missing data -> replace by mean
        "horsepower": 2 missing data -> replace by mean
        "peak_rpm": 2 missing data -> replace by mean
        "price": 4 missing data -> drop the whole row as wrong data might affect the prediction
        '''
        print("number of NaN values for the column normalized_losses :", self.df["normalized_losses"].isnull().sum())
        print("number of NaN values for the column price :", self.df['price'].isnull().sum())
        missing_data = self.df.isnull()
        # print(missing_data.columns.values.tolist())

        # from below code we can find out which column has how many missing data
        # for column in missing_data.columns.values.tolist():
        #     print(column)
        #     print (missing_data[column].value_counts())
        #     print("")

        avg_norm_loss = self.df["normalized_losses"].astype("float").mean(axis=0)
        print("Average of normalized_losses:", avg_norm_loss)
        self.df.replace({"normalized_losses": {np.nan: avg_norm_loss}}, inplace=True)

        avg_bore=self.df['bore'].astype('float').mean(axis=0)
        print("Average of bore:", avg_bore)
        self.df['bore'] = self.df['bore'].replace(np.nan, avg_bore)

        # self.df["num_of_doors"].value_counts() # returns a dataframe with repeated data count
        door_data = self.df["num_of_doors"].value_counts().idxmax()
        self.df["num_of_doors"] = self.df["num_of_doors"].replace(np.nan, door_data)

        avg_stroke = self.df["stroke"].astype("float").mean(axis=0)
        print("Average of stroke:", avg_stroke)
        self.df.replace({"stroke": {np.nan: avg_stroke}}, inplace=True)

        avg_horsepower = self.df["horsepower"].astype("float").mean(axis=0)
        print("Average of horsepower:", avg_horsepower)
        self.df.replace({"horsepower": {np.nan: avg_horsepower}}, inplace=True)

        avg_peak_rpm = self.df["peak_rpm"].astype("float").mean(axis=0)
        print("Average of peak_rpm:", avg_peak_rpm)
        self.df.replace({"peak_rpm": {np.nan: avg_peak_rpm}}, inplace=True)

        # drop the missing value in price column
        self.df.dropna(subset=["price"], axis=0, inplace=True) # drops the rows which has NaN data axis=1 drops the entire column

    def convert_to_proper_datatype(self):
        '''
        Data cleaning Step4
        data cleaning is checking and making sure that all data is 
        in the correct format (int, float, text or other).
        '''

        self.df[["bore", "stroke", "price", 
                 "peak_rpm", "horsepower"]] = self.df[["bore", "stroke", "price", 
                                                       "peak_rpm", "horsepower"]].astype("float")
        self.df[["normalized_losses"]] = self.df[["normalized_losses"]].astype("int")

        # unnique count provides the no of unique value in each column
        unique_counts = pd.DataFrame.from_records(
            [(col, self.df[col].nunique()) for col in self.df.columns],
            columns=['Column_Name', 'Num_Unique']).sort_values(by=['Num_Unique'])

        print(f"unique data in each column: \n {unique_counts}")

        # aspiration has std and turbo categorial data
        # lets convert it to category datatype, so that it would perform fast
        # and will use less memory

        self.df["aspiration"] = self.df["aspiration"].astype("category")
        print(f"Data types assigned to pandas df={self.df.dtypes}")

    def car_data_standardization(self):
        '''
        Applying calculation to an entire column (from mpg to L/100km)
        apply data transformation to transform mpg into L/100km.
        '''

        self.df["city_mpg"] = 235/self.df["city_mpg"]
        self.df.rename(columns={"city_mpg": "city_L/100km"}, inplace=True)

        self.df["highway_mpg"] = 235/self.df["highway_mpg"]
        self.df.rename(columns={"highway_mpg": "highway_L/100km"}, inplace=True)

        print(self.df.head())


    def car_data_normalization(self):
        """
        Data Normalization
        Normalize those variables so their value ranges from 0 to 1

        one easy way by using Pandas: (here I want to use mean normalization)
        normalized_df=(df-df.mean())/df.std()
        
        to use min-max normalization:
        normalized_df=(df-df.min())/(df.max()-df.min())
        
        """
        # print(self.df[["length", "width", "height"]])
        # Simple Feature Scaling
        # self.df["length"] = self.df["length"]/self.df["length"].max()
        # self.df["width"] = self.df["width"]/self.df["width"].max()
        # self.df["height"] = self.df["height"]/self.df["height"].max()
        self.df["width"] = (self.df["width"]-self.df["width"].min())/(self.df["width"].max()-self.df["width"].min()) # MIN MAX Method
        self.df["height"] = (self.df["height"]-self.df["height"].min())/(self.df["height"].max()-self.df["height"].min())
        self.df["length"] = (self.df["length"]-self.df["length"].min())/(self.df["length"].max()-self.df["length"].min())
        # self.df["height"] = (self.df["height"]- self.df["height"].mean())/self.df["height"].std()

        # print(self.df[["length", "width", "height"]].head()) # top 10 rows for particular column

    def car_data_binning(self):
        '''
        Grouping values into bins, creating some range bucket and keeping the data in bins
        linspace(start_value, end_value, numbers_generated)
        '''
        bins = np.linspace(min(self.df["horsepower"]), max(self.df["horsepower"]), 4)

        group_names = ['Low', 'Medium', 'High']
        self.df['horsepower-binned'] = pd.cut(self.df['horsepower'], bins, labels=group_names, include_lowest=True )
        print(self.df[['horsepower','horsepower-binned']].head())
        self.create_bar_chart_out_of_categorial(group_names, "horsepower-binned",
                                                xlabel="horsepower", ylabel="count",
                                                title="horsepower bins")
        
        body_style_df = self.df['body-style'].value_counts() # this is a Series
        print(body_style_df.info())
    
        
    def create_bar_chart_out_of_categorial(self, groups, col, **kwargs):
        """plot a histogram our of categorial data"""
        
        pyplot.bar(groups, self.df[col].value_counts())
        xlabel = kwargs.get('xlabel')
        ylabel = kwargs.get('ylabel')
        title = kwargs.get('title')

        # set x/y labels and plot title
        pyplot.xlabel(xlabel)
        pyplot.ylabel(ylabel)
        pyplot.title(title)

        pyplot.show()

    def categorial_to_indicative(self):
        '''
        convert the categorical variables to columns
        e.g. fuel_type col has gas and diesel so gas and diesel becomes col with True/False data 
        '''
        dummy_variable_1 = pd.get_dummies(self.df["fuel_type"])

        dummy_variable_1.rename(
            columns={'gas':'fuel-type-gas', 'diesel':'fuel-type-diesel'}, inplace=True)

        # merge data frame "df" and "dummy_variable_1"
        self.df = pd.concat([self.df, dummy_variable_1], axis=1)
        print(self.df.head(), self.df.shape)
        # print(f"Data info after cleaning={self.df.info()}")

    def misellenious_correlation(self):
        '''
        Helps to find the correlation between the columns in the dataframe
        it cannot find the correrelation between string object datatype
        '''
        # Select only numeric columns for correlation
        numeric_df = self.df.select_dtypes(include=['float64', 'int64'])
        result_df = numeric_df.corr()
        # print(result_df)

        # Find the correlation between the following columns: bore, stroke, compression-ratio, and horsepower.
        corr_between_some_imp_col = self.df[['bore','stroke',
                                             'compression-ratio','horsepower', 'engine_size', 'price']].corr()
        print(corr_between_some_imp_col)

        # computes pearson correlation value
        corre = self.df['horsepower'].corr(self.df['price'])
        print(f"correrelation value of horsepower and price={corre}")

        # can apply the method "describe" on the variables of type 'object' 
        self.df.describe(include=['object'])

if __name__== '__main__':
    ds = ImportCleaningDataSet()
    ds.reading_car_dataset()
    ds.identify_handle_missing_data()
    ds.evaluate_and_deal_missing_data()
    ds.convert_to_proper_datatype()
    ds.car_data_standardization()
    ds.car_data_normalization()
    ds.car_data_binning()
    ds.categorial_to_indicative()
    ds.misellenious_correlation()
    ds.save_the_modified_dataframe()
