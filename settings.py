'''
setting file keeps all the environment variables
'''
import os
from dotenv import load_dotenv
load_dotenv()

DBNAME = os.getenv("DBNAME")
USERNAME = os.getenv("USER")
PASSWORD = os.getenv("PASSWORD")
HOST = os.getenv("HOST")
PORT = os.getenv("PORT")
