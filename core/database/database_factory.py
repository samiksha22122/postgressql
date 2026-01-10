from core.database.database_client import DatabaseClient

class DatabaseFactory:
    def __init__(self, database_service: DatabaseClient):
        self.database_service = database_service

    def read_query(self, query: str):
        return self.database_service.read_query(query)
    
    def fetch(self, query: str):
        return self.database_service.fetch(query)

    def insert(self, query: str, values: tuple):
        return self.database_service.insert(query, values)

    def update(self, query: str, values: tuple):
        return self.database_service.update(query, values)
    
    def delete(self, query: str, values: tuple = None):
        return self.database_service.delete(query, values)
