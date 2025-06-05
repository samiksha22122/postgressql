from abc import ABC, abstractmethod

class DatabaseClient(ABC):
    @abstractmethod
    def connect(self):
        pass

    @abstractmethod
    def read_query(self, query: str):
        pass

    @abstractmethod
    def fetch(self, query: str):
        pass

    @abstractmethod
    def insert(self, query: str, values: tuple):
        pass

    @abstractmethod
    def update(self, query: str, values: tuple):
        pass

    @abstractmethod
    def delete(self, query: str, values: tuple = None):
        pass
