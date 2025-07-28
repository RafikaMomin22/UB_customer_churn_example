import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('churn_prediction.log'),
        logging.StreamHandler()
    ]
)

# File paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data' / 'inputs'  # static files
POST_DIR = BASE_DIR / 'post'  # output directory

# Data configuration
DATA_FILE = 'online_retail_customer_churn.csv'
TARGET_COL = 'Target_Churn'

# Model configuration
TEST_SIZE = 0.2
RANDOM_STATE = 42
IMBALANCE_THRESHOLD = 0.10  # Use SMOTE if churn rate ≤ 10%
CORRELATION_THRESHOLD = 0.8  # Updated correlation threshold
NUM_CV_FOLDS = 5  # Number of cross-validation folds
CV_SCORING = 'f1'  # Scoring metric for cross-validation
SHAP_SAMPLE_SIZE = 1000  # Number of samples to use for SHAP analysis
SHAP_BACKGROUND_SIZE = 100  # Samples for KernelExplainer
