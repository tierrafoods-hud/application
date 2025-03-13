import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# migration file for models table
from config.database import get_db
from dotenv import load_dotenv

load_dotenv()

db_type = os.getenv('DB_TYPE')
db = get_db(db_type)

if db_type == 'mysql':
    query = """
    CREATE TABLE IF NOT EXISTS `models` (
        `id` INT NOT NULL AUTO_INCREMENT,
        `title` VARCHAR(100) NOT NULL,
        `description` TEXT NULL,
        `model_type` VARCHAR(100) NOT NULL,
        `features` JSON NOT NULL,
        `target` VARCHAR(100) NOT NULL,
        `metrics` JSON NULL,
        `path` TEXT NOT NULL,
        `last_updated` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (`id`)
    );
    """
elif db_type == 'postgresql':
    query = """
    CREATE TABLE IF NOT EXISTS models (
        id SERIAL PRIMARY KEY,
        title VARCHAR(100) NOT NULL,
        description TEXT NULL,
        model_type VARCHAR(100) NOT NULL,
        features JSONB NOT NULL,
        target VARCHAR(100) NOT NULL,
        metrics JSONB NULL,
        path TEXT NOT NULL,
        last_updated TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
    );
    """
db.create(query)
db.close()