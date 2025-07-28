import pytest
import pandas as pd
import numpy as np
import logging
from sklearn.datasets import make_classification
from churn_model.prep.datasource import DataSource
from churn_model.prep.dataprep import DataPrep
from churn_model.model.model import ChurnModel


@pytest.fixture
def sample_data():
    # Create a synthetic dataset for testing
    X, y = make_classification(
        n_samples=100,
        n_features=10,
        n_classes=2,
        weights=[0.9, 0.1],  # Imbalanced
        random_state=42
    )

    # Create DataFrame with meaningful columns
    df = pd.DataFrame(X, columns=[
        'Age', 'Annual_Income', 'Total_Spend', 'Years_as_Customer',
        'Num_of_Purchases', 'Average_Transaction_Amount',
        'Num_of_Returns', 'Num_of_Support_Contacts',
        'Satisfaction_Score', 'Last_Purchase_Days_Ago'
    ])

    # Add categorical features
    df['Gender'] = np.random.choice(['Male', 'Female', 'Other'], size=100)
    df['Email_Opt_In'] = np.random.choice([True, False], size=100)
    df['Promotion_Response'] = np.random.choice(['Responded', 'Ignored', 'Unsubscribed'], size=100)
    df['Target_Churn'] = y

    return df


def test_data_loading(tmp_path, sample_data):
    # Save sample data to temp file
    file_path = tmp_path / "test_data.csv"
    sample_data.to_csv(file_path, index=False)

    # Test DataSource
    ds = DataSource()
    ds.file_path = file_path
    assert ds.load_data() == True
    assert ds.get_data() is not None
    assert 'Target_Churn' in ds.get_data().columns


def test_data_preprocessing(sample_data):
    # Test DataPrep
    dp = DataPrep(sample_data)
    assert dp.preprocess_data() == True

    X_train, X_test, y_train, y_test = dp.get_train_test_data()
    assert X_train is not None
    assert X_test is not None
    assert y_train is not None
    assert y_test is not None
    assert len(X_train) + len(X_test) == len(sample_data)

    # Check if imbalance was detected
    assert dp.get_imbalance_level() in ['severe', 'high', 'moderate', 'mild', 'balanced']


def test_model_training(sample_data):
    dp = DataPrep(sample_data)
    dp.preprocess_data()
    X_train, _, y_train, _ = dp.get_train_test_data()

    model = ChurnModel(dp.get_imbalance_level())
    model.initialize_models()
    model.train_models(X_train, y_train)

    assert model.best_model is not None
    assert model.best_score > 0


def test_model_evaluation(sample_data):
    dp = DataPrep(sample_data)
    dp.preprocess_data()
    X_train, X_test, y_train, y_test = dp.get_train_test_data()

    model = ChurnModel(dp.get_imbalance_level())
    model.train_models(X_train, y_train)

    metrics = model.evaluate_model(X_test, y_test)
    assert 'accuracy' in metrics
    assert 0 <= metrics['accuracy'] <= 1


def test_model_saving_loading(sample_data, tmp_path):
    dp = DataPrep(sample_data)
    dp.preprocess_data()
    X_train, _, y_train, _ = dp.get_train_test_data()

    model = ChurnModel(dp.get_imbalance_level())
    model.train_models(X_train, y_train)

    model_path = tmp_path / "test_model.pkl"
    assert model.save_model(model_path) == True

    # Test that we can load the model (would need to modify class to add load method)
    # This is just a placeholder test
    assert model_path.exists()