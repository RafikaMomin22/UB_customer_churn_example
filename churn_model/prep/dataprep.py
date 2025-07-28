import pandas as pd
import numpy as np
import logging
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
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

    def _analyze_missing_values(self, df):
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

    def _handle_outliers(self, df):
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

    def _analyze_correlations(self, df):
        """Analyze correlations between numerical features"""
        try:
            # Ensure output directory exists
            POST_DIR.mkdir(exist_ok=True)

            # Select only numerical features
            numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
            numerical_cols = [col for col in numerical_cols if col != TARGET_COL]

            if len(numerical_cols) < 2:
                logging.warning("Not enough numerical features for correlation analysis")
                return

            # Calculate correlation matrix
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
                    if abs(corr_matrix.iloc[i, j]) > CORRELATION_THRESHOLD:
                        high_corr.append({
                            'feature1': corr_matrix.columns[i],
                            'feature2': corr_matrix.columns[j],
                            'correlation': round(corr_matrix.iloc[i, j], 3)
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
            plt.title("Feature Correlation Matrix")
            plot_path = POST_DIR / 'feature_correlations.png'
            plt.savefig(plot_path)
            plt.close()
            logging.info(f"Saved correlation matrix plot to {plot_path}")

        except Exception as e:
            logging.error(f"Correlation analysis failed: {str(e)}", exc_info=True)

    def _check_imbalance(self):
        """Check if we need to handle imbalance with SMOTE"""
        churn_rate = self.data[TARGET_COL].mean()
        self.use_smote = churn_rate <= IMBALANCE_THRESHOLD
        logging.info(f"Churn rate: {churn_rate:.2%} | SMOTE {'enabled' if self.use_smote else 'disabled'}")

    def _create_preprocessor(self, X):
        """Create preprocessing pipeline with proper feature naming"""
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'bool']).columns.tolist()

        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        return ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)],
            verbose_feature_names_out=False  # Cleaner feature names
        )

    def preprocess_data(self):
        """Complete preprocessing pipeline with proper ordering"""
        try:
            logging.info("Starting data preprocessing...")
            df = self.data.copy()

            # Step 1: Convert target to numeric
            df[TARGET_COL] = df[TARGET_COL].astype(int)

            # Step 2: Handle missing values
            df = self._analyze_missing_values(df)

            # Step 3: Handle outliers
            df = self._handle_outliers(df)

            # Step 4: Check class imbalance
            self._check_imbalance()

            # Step 5: Analyze feature correlations
            self._analyze_correlations(df)

            # Define features and target
            X = df.drop(TARGET_COL, axis=1)
            y = df[TARGET_COL]

            # Step 6: Create preprocessor
            self.preprocessor = self._create_preprocessor(X)

            # Step 7: Split data (80/20)
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y,
                test_size=TEST_SIZE,
                random_state=RANDOM_STATE,
                stratify=y
            )

            # Step 8: Apply SMOTE if needed (only on training data)
            if self.use_smote:
                logging.info("Applying SMOTE for class imbalance")
                smote = SMOTE(random_state=RANDOM_STATE)
                self.X_train, self.y_train = smote.fit_resample(
                    self.preprocessor.fit_transform(self.X_train),
                    self.y_train
                )
            else:
                self.X_train = self.preprocessor.fit_transform(self.X_train)

            # Transform test data (no fitting)
            self.X_test = self.preprocessor.transform(self.X_test)

            logging.info("Data preprocessing completed successfully")
            return True

        except Exception as e:
            logging.error(f"Preprocessing failed: {str(e)}", exc_info=True)
            return False

    def get_all_data(self):
        """Return all preprocessed data before train/test split"""
        df = self.data.copy()
        df[TARGET_COL] = df[TARGET_COL].astype(int)
        return df.drop(TARGET_COL, axis=1), df[TARGET_COL]

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

    def get_train_test_data(self):
        """Return preprocessed train and test data"""
        return self.X_train, self.X_test, self.y_train, self.y_test

    def get_missing_values_report(self):
        """Return the missing values analysis report (numerical only)"""
        return self.missing_values_report

    def get_outliers_report(self):
        """Return the outliers treatment report"""
        return self.outliers_report

    def get_correlation_report(self):
        """Return the feature correlation report"""
        return self.correlation_report
