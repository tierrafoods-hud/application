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
        `target` VARCHAR(100) NOT NULL,
        `feature_columns` JSON NOT NULL,
        `categorical_features` JSON NULL,
        `numeric_features` JSON NULL,
        `metrics` JSON NULL,
        `pipeline_path` TEXT NOT NULL,
        `training_config` JSON NULL,
        `data_shape` JSON NULL,
        `created_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        `updated_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        `version` VARCHAR(50) DEFAULT '1.0',
        `is_active` BOOLEAN DEFAULT TRUE,
        PRIMARY KEY (`id`),
        INDEX `idx_model_type` (`model_type`),
        INDEX `idx_target` (`target`),
        INDEX `idx_active` (`is_active`)
    );
    """
elif db_type == 'postgresql':
    query = """
    CREATE TABLE IF NOT EXISTS models (
        id SERIAL PRIMARY KEY,
        title VARCHAR(100) NOT NULL,
        description TEXT NULL,
        model_type VARCHAR(100) NOT NULL,
        target VARCHAR(100) NOT NULL,
        feature_columns JSONB NOT NULL,
        categorical_features JSONB NULL,
        numeric_features JSONB NULL,
        metrics JSONB NULL,
        pipeline_path TEXT NOT NULL,
        training_config JSONB NULL,
        data_shape JSONB NULL,
        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        version VARCHAR(50) DEFAULT '1.0',
        is_active BOOLEAN DEFAULT TRUE
    );
    CREATE INDEX IF NOT EXISTS idx_model_type ON models(model_type);
    CREATE INDEX IF NOT EXISTS idx_target ON models(target);
    CREATE INDEX IF NOT EXISTS idx_active ON models(is_active);
    """
try:
    db.create(query)
    print(f"{db_type.capitalize()} models table created successfully")
except Exception as e:
    print(f"Failed to create {db_type.capitalize()} models table: {str(e)}")
finally:
    db.close()