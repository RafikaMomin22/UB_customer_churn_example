import pytest
import pandas as pd
from pathlib import Path
from churn_model.config import DATA_DIR, DATA_FILE
from churn_model.prep.datasource import DataSource
import logging

# Setup test data path
TEST_DATA_PATH = DATA_DIR / DATA_FILE


@pytest.fixture
def load_raw_data():
    """Fixture to load raw data once for all tests"""
    datasource = DataSource()
    datasource.load_data()
    return datasource.get_data()


def test_data_file_exists():
    """Check that the input data file exists"""
    assert TEST_DATA_PATH.exists(), f"Data file not found at {TEST_DATA_PATH}"


def test_required_columns_present(load_raw_data):
    """Verify all expected columns exist in the dataset"""
    required_columns = {
        'Age', 'Years_as_Customer', 'Satisfaction_Score',
        'Annual_Income', 'Target_Churn'  # Add other required columns
    }
    assert required_columns.issubset(load_raw_data.columns), \
        f"Missing columns: {required_columns - set(load_raw_data.columns)}"


def test_age_validation(load_raw_data):
    """Check Age values are between 18 and 110"""
    age_series = load_raw_data['Age']
    invalid_ages = age_series[(age_series < 18) | (age_series > 110)]

    assert invalid_ages.empty, \
        f"Found {len(invalid_ages)} invalid Age values:\n{invalid_ages.head()}"


def test_years_as_customer_validation(load_raw_data):
    """Check Years_as_Customer values are between 0 and 110"""
    years_series = load_raw_data['Years_as_Customer']
    invalid_years = years_series[(years_series < 0) | (years_series > 110)]

    assert invalid_years.empty, \
        f"Found {len(invalid_years)} invalid Years_as_Customer values:\n{invalid_years.head()}"


def test_satisfaction_score_validation(load_raw_data):
    """Check Satisfaction_Score is between 0 and 5 (inclusive)"""
    score_series = load_raw_data['Satisfaction_Score']
    invalid_scores = score_series[~score_series.isin(range(0, 6))]  # 0-5 inclusive

    assert invalid_scores.empty, \
        f"Found {len(invalid_scores)} invalid Satisfaction_Score values:\n{invalid_scores.head()}"


def test_no_missing_target(load_raw_data):
    """Check that target column has no missing values"""
    assert load_raw_data['Target_Churn'].notna().all(), \
        "Found missing values in Target_Churn"


