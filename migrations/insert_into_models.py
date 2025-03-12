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
    INSERT INTO `models` (`title`, `description`, `model_type`, `features`, `target`, `scaler`, `metrics`, `path`) VALUES
    ('(UoH) Mexico orgc', 'Model trained by the University of Huddersfield on for Mexico Location to predict organic carbon', 'Linear Regression', '[\"upper_depth\", \"lower_depth\", \"elcosp\", \"phaq\", \"sand\", \"temperature\", \"precipitation\", \"landcover\", \"silt_plus_clay\", \"organic_matter\", \"bulk_density\"]', 'orgc', 'models/UoH/Mexico_orgc_scaler.pkl', '{\"mse\": 12.988519362186787, \"mae\": 2.3849658314350797, \"r2\": 0.9042858368636583}', 'models/UoH/Mexico_orgc_best_model.pkl'),
    ('(UoH) United Kingdom orgc', 'Model trained by the University of Huddersfield on for United Kingdom Location to predict organic carbon', 'Linear Regression', '[\"upper_depth\", \"lower_depth\", \"phaq\", \"sand\", \"silt_plus_clay\", \"organic_matter\", \"bulk_density\"]', 'orgc', 'models/UoH/UK_orgc_scaler.pkl', '{\"mse\": 6.786195965417868, \"mae\": 0.32662824207492797, \"r2\": 0.9989300768411852}', 'models/UoH/UK_orgc_best_model.pkl');
    """
elif db_type == 'postgresql':
    query = """
    INSERT INTO `models` (`title`, `description`, `model_type`, `features`, `target`, `scaler`, `metrics`, `path`) VALUES
    ('(UoH) Mexico orgc', 'Model trained by the University of Huddersfield on for Mexico Location to predict organic carbon', 'Linear Regression', '[\"upper_depth\", \"lower_depth\", \"elcosp\", \"phaq\", \"sand\", \"temperature\", \"precipitation\", \"landcover\", \"silt_plus_clay\", \"organic_matter\", \"bulk_density\"]', 'orgc', 'models/UoH/Mexico_orgc_scaler.pkl', '{\"mse\": 12.988519362186787, \"mae\": 2.3849658314350797, \"r2\": 0.9042858368636583}', 'models/UoH/Mexico_orgc_best_model.pkl'),
    ('(UoH) United Kingdom orgc', 'Model trained by the University of Huddersfield on for United Kingdom Location to predict organic carbon', 'Linear Regression', '[\"upper_depth\", \"lower_depth\", \"phaq\", \"sand\", \"silt_plus_clay\", \"organic_matter\", \"bulk_density\"]', 'orgc', 'models/UoH/UK_orgc_scaler.pkl', '{\"mse\": 6.786195965417868, \"mae\": 0.32662824207492797, \"r2\": 0.9989300768411852}', 'models/UoH/UK_orgc_best_model.pkl');
    """

db.create(query)
db.close()