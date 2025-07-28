import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import logging
import pickle
import json
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold)
from sklearn.pipeline import Pipeline
from churn_model.config import (POST_DIR, RANDOM_STATE,
                                NUM_CV_FOLDS, CV_SCORING)


class ChurnModel:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_score = 0
        self.best_model_name = ""
        self.preprocessor = None
        self.cv_scores = {}
        self.param_grids = {}

    def initialize_models(self):
        """Initialize models with parameter grids for tuning"""
        logging.info("Initializing models with parameter grids")

        self.param_grids = {
            'Logistic Regression': {
                'classifier__C': [0.1, 1, 10],
                'classifier__penalty': ['l2'],
                'classifier__solver': ['lbfgs', 'liblinear']
            },
            'Random Forest': {
                'classifier__n_estimators': [100, 200],
                'classifier__max_depth': [None, 10, 20],
                'classifier__min_samples_split': [2, 5]
            },
            'Gradient Boosting': {
                'classifier__n_estimators': [100, 200],
                'classifier__learning_rate': [0.01, 0.1],
                'classifier__max_depth': [3, 5]
            },
            'SVM': {
                'classifier__C': [0.1, 1, 10],
                'classifier__kernel': ['linear', 'rbf'],
                'classifier__gamma': ['scale', 'auto']
            }
        }

        self.models = {
            'Logistic Regression': LogisticRegression(
                class_weight='balanced',
                max_iter=1000,
                random_state=RANDOM_STATE
            ),
            'Random Forest': RandomForestClassifier(
                class_weight='balanced',
                random_state=RANDOM_STATE
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                random_state=RANDOM_STATE
            ),
            'SVM': SVC(
                class_weight='balanced',
                probability=True,
                random_state=RANDOM_STATE
            )
        }

    def train_with_cv(self, X_train, y_train, preprocessor):
        """
        Train models using GridSearchCV with F1-based evaluation.
        Args:
            X_train: Preprocessed training features (numpy array)
            y_train: Training labels
            preprocessor: Fitted ColumnTransformer for feature names
        Returns:
            Best trained model (Pipeline)
        """
        if not self.models:
            self.initialize_models()

        self.preprocessor = preprocessor
        cv = StratifiedKFold(n_splits=NUM_CV_FOLDS,
                             shuffle=True,
                             random_state=RANDOM_STATE)

        for name, model in self.models.items():
            try:
                logging.info(f"\nTraining {name} with CV...")

                pipeline = Pipeline([('classifier', model)])

                grid_search = GridSearchCV(
                    estimator=pipeline,
                    param_grid=self.param_grids[name],
                    cv=cv,
                    scoring=CV_SCORING,  # Uses F1 from config
                    n_jobs=-1,
                    verbose=1
                )

                grid_search.fit(X_train, y_train)

                self.cv_scores[name] = {
                    'best_score': grid_search.best_score_,
                    'best_params': grid_search.best_params_,
                    'cv_results': grid_search.cv_results_
                }

                logging.info(f"Best F1: {grid_search.best_score_:.4f}")

                if grid_search.best_score_ > self.best_score:
                    self.best_score = grid_search.best_score_
                    self.best_model = grid_search.best_estimator_
                    self.best_model_name = name

            except Exception as e:
                logging.error(f"CV failed for {name}: {str(e)}")

        logging.info(f"\nSelected {self.best_model_name} (F1={self.best_score:.4f})")
        return self.best_model

    def evaluate_model(self, X_test, y_test):
        """Evaluate best model on test data using multiple metrics"""
        if not self.best_model:
            raise ValueError("No trained model available")

        y_pred = self.best_model.predict(X_test)
        y_proba = self.best_model.predict_proba(X_test)[:, 1]

        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),  # Primary metric
            'roc_auc': roc_auc_score(y_test, y_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }

    def _analyze_shap(self, X_test, model):
        """Generate SHAP explanations (test data only)"""
        try:
            if isinstance(model, (RandomForestClassifier, GradientBoostingClassifier)):
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_test)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]  # Use positive class
            elif isinstance(model, LogisticRegression):
                explainer = shap.LinearExplainer(model, X_test)
                shap_values = explainer.shap_values(X_test)
            else:
                logging.warning(f"SHAP not supported for {type(model).__name__}")
                return None

            # Save plots and feature importance
            self._save_shap_outputs(X_test, shap_values, explainer)
            return pd.DataFrame({
                'feature': X_test.columns,
                'shap_importance': np.abs(shap_values).mean(axis=0)
            }).sort_values('shap_importance', ascending=False).head(5)

        except Exception as e:
            logging.error(f"SHAP failed: {str(e)}")
            return None

    def _save_shap_outputs(self, X_test, shap_values, explainer):
        """Save SHAP plots and data"""
        POST_DIR.mkdir(exist_ok=True)

        # Summary plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test, show=False)
        plt.savefig(POST_DIR / 'shap_summary.png', bbox_inches='tight')
        plt.close()

    def save_model(self, file_name='churn_model.pkl'):
        """Save best model to disk"""
        if not self.best_model:
            raise ValueError("No model to save")

        POST_DIR.mkdir(exist_ok=True)
        with open(POST_DIR / file_name, 'wb') as f:
            pickle.dump({
                'model': self.best_model,
                'preprocessor': self.preprocessor,
                'metadata': {
                    'best_score': self.best_score,
                    'best_model_name': self.best_model_name
                }
            }, f)
        logging.info(f"Model saved to {POST_DIR / file_name}")

    def save_metrics(self, metrics, file_name='metrics.json'):
        """Save evaluation metrics"""
        POST_DIR.mkdir(exist_ok=True)
        with open(POST_DIR / file_name, 'w') as f:
            json.dump(metrics, f, indent=4)
        logging.info(f"Metrics saved to {POST_DIR / file_name}")
