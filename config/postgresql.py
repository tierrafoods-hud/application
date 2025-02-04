# for postgresql
import psycopg2
from dotenv import load_dotenv
import os

load_dotenv()

DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_NAME = os.getenv('DB_NAME')

if not DB_HOST or not DB_PORT or not DB_USER or not DB_NAME:
    raise ValueError("Missing PostgreSQL environment variables")

class DB:

    def __init__(self):
        self.connection = self.get_connection()
        self.cursor = self.get_cursor()

    def get_connection(self):
        return psycopg2.connect(host=DB_HOST, port=DB_PORT, user=DB_USER, password=DB_PASSWORD, dbname=DB_NAME)

    def get_cursor(self):
        return self.connection.cursor()

    def create(self, query, params=()):
        self.cursor.execute(query, params)
        self.connection.commit()
        return self.cursor.rowcount

    def fetchAll(self, query, params=()):
        self.cursor.execute(query, params)
        return self.cursor.fetchall()
    
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
        self.connection.close()
        self.cursor.close()