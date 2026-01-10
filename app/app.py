# app.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

from core.database.postgres_client import PostgresClient
from core.database.database_factory import DatabaseFactory

# Database configuration
db_config = {
    "dbname": "ipro_db",
    "user": "supplychain",
    "password": "supplychain",
    "host": "localhost",
    "port": 5432
}

# Initialize database clients
postgres_client = PostgresClient(db_config)
db_factory = DatabaseFactory(postgres_client)

# Initialize FastAPI app
app = FastAPI(title="Employee Management API")


# -----------------------------
# Request Models
# -----------------------------
class Employee(BaseModel):
    id: int
    name: str
    department: str
    salary: float


class SalaryUpdate(BaseModel):
    name: str
    salary: float


# -----------------------------
# API Endpoints
# -----------------------------

@app.post("/employee", response_model=dict)
def add_employee(employee: Employee):
    """Insert a new employee into the database."""
    try:
        db_factory.insert(
            "INSERT INTO employee (id, name, department, salary) VALUES (%s, %s, %s, %s);",
            (employee.id, employee.name, employee.department, employee.salary)
        )
        return {"message": "Employee added successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/employee/salary", response_model=dict)
def update_salary(salary_update: SalaryUpdate):
    """Update an employee's salary by name."""
    try:
        db_factory.update(
            "UPDATE employee SET salary = %s WHERE name = %s;",
            (salary_update.salary, salary_update.name)
        )
        return {"message": "Salary updated successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/employee/cleanup", response_model=dict)
def cleanup_null_ids():
    """Delete employees where ID is NULL."""
    try:
        db_factory.delete("DELETE FROM employee WHERE id IS NULL;")
        return {"message": "Cleanup completed."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/employee", response_model=List[dict])
def get_employee_by_name(name: str):
    """Retrieve employee data by name."""
    try:
        df = db_factory.read_query("SELECT * FROM employee WHERE name = %s;", (name,))
        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/employee/all", response_model=List[dict])
def get_all_employees():
    """Fetch all employees ordered by ID."""
    try:
        rows = db_factory.fetch("SELECT * FROM employee ORDER BY id ASC;")
        return rows
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
