import numpy as np
import pandas as pd
import logging
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from churn_model.config import (TARGET_COL, TEST_SIZE, RANDOM_STATE,
                                IMBALANCE_THRESHOLD, CORRELATION_THRESHOLD,
                                POST_DIR)


class DataPrep:
    def __init__(self, data):
        self.data = data
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.preprocessor = None
        self.use_smote = False
        self.missing_values_report = None
        self.outliers_report = None
        self.correlation_report = None

    def analyze_missing_values(self, df):
        """Analyze and impute missing values in numerical columns only"""
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
        missing_values = df[numerical_cols].isnull().sum()
        missing_values = missing_values[missing_values > 0]

        if not missing_values.empty:
            self.missing_values_report = {
                'columns_with_missing': missing_values.index.tolist(),
                'missing_counts': missing_values.values.tolist(),
                'missing_percentages': (missing_values / len(df) * 100).round(2).tolist(),
                'imputation_method': 'median'
            }
            logging.warning(f"Found missing values in {len(missing_values)} numerical columns")

            # Perform actual imputation here
            for col in self.missing_values_report['columns_with_missing']:
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                logging.info(f"Imputed {col} with median value: {median_val:.2f}")

            for col, count, pct in zip(self.missing_values_report['columns_with_missing'],
                                       self.missing_values_report['missing_counts'],
                                       self.missing_values_report['missing_percentages']):
                logging.warning(f" - {col}: {count} missing values ({pct}%)")
        else:
            self.missing_values_report = {
                'columns_with_missing': [],
                'imputation_method': None
            }
            logging.info("No missing values found in numerical columns")
        return df

    def handle_outliers(self, df):
        """Detect and handle outliers in specific columns"""
        outlier_columns = ['Annual_Income', 'Total_Spend', 'Average_Transaction_Amount']
        self.outliers_report = {}

        for col in outlier_columns:
            if col not in df.columns:
                continue

            # Calculate bounds for outliers (using IQR method)
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            # Detect outliers
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            outlier_count = len(outliers)

            if outlier_count > 0:
                logging.warning(f"Found {outlier_count} outliers in {col}")

                # Apply different treatments based on column
                if col == 'Annual_Income':
                    # Apply log transformation (add 1 to avoid log(0))
                    df[col] = np.log1p(df[col])
                    treatment = 'log_transform'
                else:
                    # Cap at 99th percentile for spending-related columns
                    cap_value = df[col].quantile(0.99)
                    df[col] = np.where(df[col] > cap_value, cap_value, df[col])
                    treatment = 'capped_at_99th_percentile'

                self.outliers_report[col] = {
                    'outlier_count': outlier_count,
                    'treatment': treatment,
                    'pre_stats': {
                        'min': outliers[col].min(),
                        'max': outliers[col].max(),
                        'mean': outliers[col].mean()
                    }
                }
            else:
                logging.info(f"No outliers detected in {col}")

        return df

    def check_imbalance(self, y_train):
        """Check if we need to handle imbalance with SMOTE"""
        churn_rate = y_train.mean()
        self.use_smote = churn_rate <= IMBALANCE_THRESHOLD
        logging.info(f"Churn rate: {churn_rate:.2%} | SMOTE {'enabled' if self.use_smote else 'disabled'}")

    def analyze_correlations(self, x_train_transformed, feature_names):
        """Analyze correlations between numerical features after preprocessing"""
        df = pd.DataFrame(x_train_transformed, columns=feature_names)

        # Robust exclusion of one-hot encoded features
        numerical_cols = [
            col for col in df.columns
            # Exclude any column that looks like one-hot encoded:
            if not (col.startswith(('cat__', 'bool__')) or  # sklearn default pattern
                    ('_' in col and col.split('_')[-1] in ['True', 'False', 'Yes', 'No']))]  # common binary encodings

        if len(numerical_cols) < 2:
            logging.warning("Not enough numerical features for correlation analysis")
            return

        corr_matrix = df[numerical_cols].corr()
        self.correlation_report = {
            'high_correlations': [],
            'correlation_matrix': corr_matrix.to_dict(),
            'threshold_used': CORRELATION_THRESHOLD
        }

        # Find high correlations (absolute value > threshold)
        high_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) > CORRELATION_THRESHOLD:
                    high_corr.append({
                        'feature1': corr_matrix.columns[i],
                        'feature2': corr_matrix.columns[j],
                        'correlation': round(corr_value, 3)
                    })

        self.correlation_report['high_correlations'] = high_corr

        # Log results
        if high_corr:
            logging.warning(f"High correlations (> {CORRELATION_THRESHOLD}) found between features:")
            for item in high_corr:
                logging.warning(f" - {item['feature1']} & {item['feature2']}: {item['correlation']}")
        else:
            logging.info(f"No high correlations (> {CORRELATION_THRESHOLD}) between numerical features")

        # Visualize correlation matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm',
                    center=0, vmin=-1, vmax=1)
        plt.title("Feature Correlation Matrix (Original Numerical Features)")
        plot_path = POST_DIR / 'feature_correlations.png'
        plt.savefig(plot_path)
        plt.close()
        logging.info(f"Saved correlation matrix plot to {plot_path}")

    @staticmethod
    def create_preprocessor(inputs_df):
        """Create preprocessing pipeline with proper feature naming"""
        numeric_features = inputs_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = inputs_df.select_dtypes(include=['object', 'bool']).columns.tolist()

        numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])

        categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore',
                                                                           sparse_output=False))])

        return ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)],
            verbose_feature_names_out=False  # Cleaner feature names
        )

    def preprocess_data(self):
        """Complete preprocessing pipeline with proper ordering"""
        logging.info("Starting data preprocessing...")
        df = self.data.copy()

        # Step 1: Convert target to numeric
        df[TARGET_COL] = df[TARGET_COL].astype(int)

        # Step 2: Split data first to prevent leakage
        inputs_df = df.drop(TARGET_COL, axis=1)
        target_df = df[TARGET_COL]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            inputs_df, target_df,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=target_df
        )

        # Step 3: Handle missing values (train and test separately)
        self.X_train = self.analyze_missing_values(self.X_train)
        self.X_test = self.analyze_missing_values(self.X_test)

        # Step 4: Handle outliers (train and test separately)
        self.X_train = self.handle_outliers(self.X_train)
        self.X_test = self.handle_outliers(self.X_test)

        # Step 5: Check class imbalance (training set only)
        self.check_imbalance(self.y_train)

        # Step 6: Create and fit preprocessor on training data only
        self.preprocessor = self.create_preprocessor(self.X_train)
        x_train_transformed = self.preprocessor.fit_transform(self.X_train)
        x_test_transformed = self.preprocessor.transform(self.X_test)

        # Step 7: Apply SMOTE if needed (only on training data)
        if self.use_smote:
            logging.info("Applying SMOTE for class imbalance")
            smote = SMOTE(random_state=RANDOM_STATE)
            x_train_transformed, self.y_train = smote.fit_resample(
                x_train_transformed,
                self.y_train
            )

        # Step 8: Analyze correlations (post-preprocessing, training data only)
        feature_names = self.get_feature_names()
        self.analyze_correlations(x_train_transformed, feature_names)

        logging.info("Data preprocessing completed successfully")
        return x_train_transformed, x_test_transformed, self.y_train, self.y_test

    def get_feature_names(self):
        """Get all feature names after preprocessing (including one-hot encoded)"""
        if not self.preprocessor:
            raise ValueError("Preprocessor not fitted yet")

        # Get numeric features (passed through as-is)
        numeric_features = self.preprocessor.transformers_[0][2]

        # Get categorical features (one-hot encoded)
        cat_transformer = self.preprocessor.transformers_[1][1].named_steps['onehot']
        categorical_features = cat_transformer.get_feature_names_out(
            self.preprocessor.transformers_[1][2]
        )

        return list(numeric_features) + list(categorical_features)
