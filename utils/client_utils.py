import logging
from importlib import import_module

from core.database.database_client import DatabaseClient
from core.utils.read_config import config_manager
from core.vectordb.vectordb_client import VectorDB


# Pass here only database config (app_database_config / domain_database_config)
def get_database_client(database_type: str):
    """
    Returns an instance of the appropriate database client based on the given database_type.

    Args:
        database_type (str): The type of the database client to retrieve. Supported values are "postgres" and "sqlite".

    Returns:
        DatabaseClient: An instance of the appropriate database client.

    Raises:
        ValueError: If an unsupported cloud service is specified.

    """
    logging.info("Loaded database configs inside utils")

    if database_type == "postgres":
        module_name = "core.clients.azure.database.postgres_client"
        class_name = "AzurePostgresClient"
    elif database_type == "sqlite":
        module_name = "core.clients.azure.database.sqlite_client"
        class_name = "SqliteClient"
    elif database_type == "hivedb":
        module_name = "core.clients.azure.database.hive_metastore_client"
        class_name = "HiveMetaStore"
    elif database_type == "mysql":
        module_name = "core.clients.azure.database.mysql_client"
        class_name = "MySQLClient"
    elif database_type == "snowflake":
        module_name = "core.clients.azure.database.snowflake_client"
        class_name = "SnowflakeClient"
    else:
        raise ValueError(f"Unsupported database type : {database_type}")
