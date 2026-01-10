from core.database.postgres_client import PostgresClient
from core.database.database_factory import DatabaseFactory

if __name__ == "__main__":
    db_config = {
        "dbname": "ipro_db",
        "user": "supplychain",
        "password": "supplychain",
        "host": "localhost",
        "port": 5432
    }

    postgres_client = PostgresClient(db_config)
    db_factory = DatabaseFactory(postgres_client)

    # Insert new employee (example)
    db_factory.insert(
        "INSERT INTO employee (id, name, department, salary) VALUES (%s, %s, %s, %s);",(10010,"Samiksha", "Engineering", 100000))


    # Update an employee salary
    db_factory.update("UPDATE employee SET salary = %s WHERE name = %s;", 
                      (500000, "Samiksha"))
    
    # Delete any row where id is NULL (cleanup)
    db_factory.delete("DELETE FROM employee WHERE id IS NULL;")

    df = db_factory.read_query("SELECT * FROM employee WHERE name = 'Samiksha';")
    print(df)

    # Fetch using cursor
    rows = db_factory.fetch("SELECT * FROM employee ORDER BY id ASC;")
    print(rows)