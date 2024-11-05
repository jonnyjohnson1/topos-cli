import psycopg2
from psycopg2 import sql, Error
from dotenv import load_dotenv, find_dotenv
import os

# Find the .env file and load environment variables
env_path = find_dotenv()
if env_path:
    print(f".env file found at: {env_path}")
    load_dotenv(env_path)
else:
    print("No .env file found.")

# Load environment variables from the .env file
load_dotenv()

# Retrieve database connection details from environment variables
host = os.getenv('POSTGRES_HOST')
database = os.getenv('POSTGRES_DB')
user = os.getenv('POSTGRES_USER')
password = os.getenv('POSTGRES_PASSWORD')
port = int(os.getenv('POSTGRES_PORT', 5432))  # Default to 5432 if not set

print("DATABASE:", database)
print(user)

try:
    # Establish a connection to the PostgreSQL database
    connection = psycopg2.connect(
        host=host,
        database=database,
        user=user,
        password=password,
        port=port
    )
    
    # Create a cursor object using the connection
    cursor = connection.cursor()
    
    # Execute a sample query to verify the connection
    cursor.execute("SELECT version();")
    
    # Fetch the result
    db_version = cursor.fetchone()
    print(f"Connected to the database. PostgreSQL version: {db_version}")
    
    # Close the cursor and connection
    cursor.close()
    
    # LIST ALL TABLES ON DATABASE
    # Create a cursor object using the connection
    cursor = connection.cursor()
    # Verify if the user was created
    cursor.execute("SELECT rolname FROM pg_roles WHERE rolname = %s;", (user,))
    user_exists = cursor.fetchone()
    
    if user_exists:
        print(f"Verification successful: User '{user}' exists in the database.")
    else:
        print(f"Verification failed: User '{user}' does not exist in the database.")
    
    # Execute a query to list all the tables in the current database schema
    cursor.execute("""
        SELECT table_schema, table_name
        FROM information_schema.tables
        WHERE table_type = 'BASE TABLE'
        AND table_schema NOT IN ('pg_catalog', 'information_schema')
        ORDER BY table_schema, table_name;
    """)
    
    tables = cursor.fetchall()

    # Print the result in a formatted way
    for table in tables:
        print(f"Schema: {table[0]}, Table Name: {table[1]}")
    
    cursor.execute("""
        SELECT table_name 
        FROM information_schema.tables
        WHERE table_schema = 'public';
    """)
    
    # Fetch and print all table names
    tables = cursor.fetchall()
    print("Tables in the database:")
    for table in tables:
        print(table[0])
        

    
    # Close the cursor and connection
    cursor.close()
    connection.close()
    print("Database connection closed.")
    
except Error as e:
    print(f"Error connecting to the database: {e}")