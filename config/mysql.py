# for mysql
import mysql.connector
from dotenv import load_dotenv
import os

load_dotenv()

DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_NAME = os.getenv('DB_NAME')

if not DB_HOST or not DB_PORT or not DB_USER or not DB_NAME:
    raise ValueError("Missing MySQL environment variables")

class DB:
    def __init__(self):

        self.connection = self.get_connection()
        self.cursor = self.get_cursor()

    def get_connection(self):
        return mysql.connector.connect(host=DB_HOST, port=DB_PORT, user=DB_USER, password=DB_PASSWORD, database=DB_NAME)

    def get_cursor(self):
        return self.connection.cursor()

    def create(self, query, params=()):
        self.cursor.execute(query, params)
        self.connection.commit()
        return self.cursor.rowcount
    
    def fetchAll(self, query, params=()):
        self.cursor.execute(query, params)
        return self.cursor.fetchall()
    
    def fetchAllAsDict(self, query, params=()):
        self.cursor.execute(query, params)
        columns = [column[0] for column in self.cursor.description]
        rows = self.cursor.fetchall()
        return [dict(zip(columns, row)) for row in rows]
    
    def fetchOne(self, query, params=()):
        self.cursor.execute(query, params)

        return self.cursor.fetchone()
    
    def fetchMany(self, query, params=(), size=1):
        self.cursor.execute(query, params)
        return self.cursor.fetchmany(size)
    
    def update(self, query, params=()):
        self.cursor.execute(query, params)
        self.connection.commit()
        return self.cursor.rowcount
    
    def delete(self, query, params=()):
        self.cursor.execute(query, params)
        self.connection.commit()
        return self.cursor.rowcount
    
    def close(self):
        self.cursor.close()
        self.connection.close()


