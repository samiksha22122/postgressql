import psycopg2
import pandas as pd
from core.database.database_client import DatabaseClient

class PostgresClient(DatabaseClient):
    def __init__(self, db_config):
        self.db_config = db_config
        self.connection = None

    def connect(self):
        self.connection = psycopg2.connect(
            dbname=self.db_config["dbname"],
            user=self.db_config["user"],
            password=self.db_config["password"],
            host=self.db_config["host"],
            port=self.db_config["port"]
        )
        self.cursor = self.connection.cursor()

    def read_query(self, query: str):
        if self.connection is None:
            self.connect()
        df = pd.read_sql_query(query, self.connection)
        return df
    
    def fetch(self, query: str):
        if self.connection is None:
            self.connect()
        self.cursor.execute(query)
        return self.cursor.fetchall()

    def insert(self, query: str, values: tuple):
        if self.connection is None:
            self.connect()
        self.cursor.execute(query, values)
        self.connection.commit()

    def update(self, query: str, values: tuple):
        if self.connection is None:
            self.connect()
        self.cursor.execute(query, values)
        self.connection.commit()

    def delete(self, query: str, values: tuple = None):
        if self.connection is None:
            self.connect()
        if values:
            self.cursor.execute(query, values)
        else:
            self.cursor.execute(query)
        self.connection.commit()
