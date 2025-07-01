
import pandas as pd
from psycopg2 import connect
from custome import DBsingletonError
import settings

class ConnectDataBase:
    '''
    This is a singleton class that ensures a class has only one instance and provide a global point of access.
    It is used for objects that need to be shared accross the entire application such as logging, database connections
    and configuration settings.
    it saves memory usage and good for development environment
    '''

    _instance = None

    @classmethod
    def get_db_connection(cls):
        if cls._instance is None:
            cls._instance = ConnectDataBase()
        return cls._instance


    def __init__(self):
        if ConnectDataBase._instance is not None:
            raise DBsingletonError("connection already established in a singleton")
        # create connection obj
        self.conn = connect(
            dbname = settings.DBNAME,
            user = settings.USERNAME,
            password = settings.PASSWORD,
            host = settings.HOST,
            port = settings.PORT
            )
        #Create a cursor object
        self.cursor = self.conn.cursor()

    def create_table(self, *args):
        '''execute create table sql command'''
        query = """CREATE TABLE instruction (
                id INT PRIMARY KEY NOT NULL,
                fname VARCHAR(15),
                lname VARCHAR(30),
                city VARCHAR(10),
                ccode CHAR(3));"""
        self.cursor.execute(query)

    def insert_into_table(self, input_data):
        '''insert data in the table'''
        for d in input_data:
            query = """INSERT INTO INSTRUCTION (ID, FNAME, LNAME, CITY, CCODE)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (ID)
                        DO NOTHING"""
            #Run queries
            self.cursor.execute(query, d)
            self.conn.commit()


    def retrieve_data(self):
        '''Retrieve data from db table'''

        query = "SELECT * FROM instruction;"
        self.cursor.execute(query)
        rs = self.cursor.fetchall()
        print(rs)

        # if want to fetch 2 rows
        # rs = self.cursor.fetchmany(2)
        # print(rs)
        # can covert as result set to df
        dataframe = pd.DataFrame(rs)
        print(dataframe)

    def retrieve_data_df(self,):
        '''retrieve data from table
        and convert to dataframe'''
        df = pd.read_sql_query(f"select * from instruction", self.conn)

        return df      

    def close_connection(self):
        '''Close the database connection
        Free resources
        '''  
        self.cursor.close()
        self.conn.close()

if __name__ == '__main__':

    try:
        db_ob1 = ConnectDataBase.get_db_connection()
        # db_obj2 = ConnectDataBase() # this raise exception
        db_obj3 = ConnectDataBase.get_db_connection()

        print(f"Check if it has created just one instances of db={db_ob1 is db_obj3}")

        data = [(1, 'Rav', 'Ahuja', 'TORONTO', 'CA'), (2, 'Raul', 'Chong', 'Markham', 'CA'), 
                (3, 'Hima', 'Vasudevan', 'Chicago', 'US')]
        db_ob1.retrieve_data()
        # db_ob1.retrieve_data_df("instruction")
    except AttributeError as e:
        print(f"DataBase ERROR={e}")
    finally:
        db_ob1.close_connection()
