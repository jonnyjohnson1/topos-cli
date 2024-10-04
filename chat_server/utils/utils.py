from datetime import datetime
import random
import string
def generate_deci_code(n_digits: int) -> str:
    return ''.join(random.choice('0123456789') for _ in range(n_digits))
def generate_group_name() -> str:
    return 'GRP-'.join(random.choices(string.ascii_uppercase + string.digits, k=8))
def sqlite_timestamp_to_ms(sqlite_timestamp: str) -> int:
    """
    Convert a SQLite timestamp string to milliseconds since Unix epoch.
    
    :param sqlite_timestamp: A timestamp string in the format "YYYY-MM-DD HH:MM:SS"
    :return: Milliseconds since Unix epoch
    """
    try:
        # Parse the SQLite timestamp string
        dt = datetime.strptime(sqlite_timestamp, "%Y-%m-%d %H:%M:%S")
        
        # Convert to milliseconds since Unix epoch
        return int(dt.timestamp() * 1000)
    except ValueError as e:
        print(f"Error parsing timestamp: {e}")
        return None