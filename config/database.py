# routing to the correct database
from .mysql import DB as MySQLDB
from .postgresql import DB as PostgreSQLDB

def get_db(db_type):
    if db_type == 'mysql':
        return MySQLDB()
    elif db_type == 'postgresql':
        return PostgreSQLDB()
    else:
        raise ValueError(f"Invalid database type: {db_type}")