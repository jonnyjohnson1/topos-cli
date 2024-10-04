
import sqlite3
def init_sqlite_database(db_file: str):
    """Initialize the SQLite database with necessary tables."""
    with sqlite3.connect(db_file) as conn:
        cursor = conn.cursor()
        
        # Create groups table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS groups (
                group_id TEXT PRIMARY KEY,
                group_name TEXT NOT NULL UNIQUE
            )
        ''')
        
        # Create users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                username TEXT NOT NULL UNIQUE,
                last_seen_online TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP    
            )
        ''')
        
        # Create user_groups relationship table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_groups (
                user_id TEXT,
                group_id TEXT,
                FOREIGN KEY (user_id) REFERENCES users (user_id),
                FOREIGN KEY (group_id) REFERENCES groups (group_id),
                PRIMARY KEY (user_id, group_id)
            )
        ''')
        
        # Optional: Add indexes for performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_user_groups_user_id ON user_groups (user_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_user_groups_group_id ON user_groups (group_id)')
        
        conn.commit()

import os

def ensure_file_exists(file_path: str) -> bool:
    """
    Check if a file exists, and create it if it doesn't.
    
    Args:
    file_path (str): The path to the file to check/create.
    
    Returns:
    bool: True if the file already existed, False if it was created.
    """
    if os.path.exists(file_path):
        return True
    else:
        try:
            # Create the directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            # Create an empty file
            open(file_path, 'a').close()
            return False
        except Exception as e:
            print(f"Error creating file {file_path}: {e}")
            return False