import pandas as pd
from pathlib import Path
from churn_model.config import DATA_DIR, DATA_FILE, logging


class DataSource:
    @staticmethod
    def load_data():
        """Load data from CSV file"""
        file_path = DATA_DIR / DATA_FILE
        logging.info(f"Loading data from {file_path}")
        orig_df = pd.read_csv(file_path)
        orig_df = orig_df.drop(columns=['Customer_ID'])  # drop customer ID since not useful for the model
        logging.info(f"Data loaded successfully. Shape: {orig_df.shape}")

        return orig_df

    # def __init__(self):
    #     self.file_path = DATA_DIR / DATA_FILE
    #     self.data = None
    #
    # def load_data(self):
    #     """Load data from CSV file"""
    #     try:
    #         logging.info(f"Loading data from {self.file_path}")
    #         self.data = pd.read_csv(self.file_path)
    #         self.data = self.data.drop(columns=['Customer_ID'])  # drop customer ID since not useful for the model
    #         logging.info(f"Data loaded successfully. Shape: {self.data.shape}")
    #         return True
    #     except Exception as e:
    #         logging.error(f"Error loading csv file: {str(e)}")
    #         return False
    #
    # def get_data(self):
    #     """Return loaded data"""
    #     return self.data
