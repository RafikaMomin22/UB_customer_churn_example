from churn_model.config import logging
from churn_model.prep.datasource import DataSource
from churn_model.prep.dataprep import DataPrep
from churn_model.model.model import ChurnModel
import pandas as pd


def main():
    try:
        logging.info("Starting churn prediction pipeline")

        # 1. Load data
        data_source = DataSource()
        if not data_source.load_data():
            raise RuntimeError("Data loading failed")

        # 2. Preprocess and split
        data_prep = DataPrep(data_source.get_data())
        if not data_prep.preprocess_data():
            raise RuntimeError("Preprocessing failed")

        X_train, X_test, y_train, y_test = data_prep.get_train_test_data()

        # 3. Train models with CV
        churn_model = ChurnModel()
        best_model = churn_model.train_with_cv(
            X_train, y_train,
            data_prep.preprocessor  # For feature names
        )

        if not best_model:
            raise RuntimeError("Model training failed")

        # 4. Evaluate on test set
        metrics = churn_model.evaluate_model(X_test, y_test)

        # 5. SHAP analysis (optional)
        try:
            X_test_df = pd.DataFrame(
                X_test,
                columns=data_prep.get_feature_names()
            )
            shap_results = churn_model._analyze_shap(
                X_test_df,
                best_model.named_steps['classifier'] if hasattr(best_model, 'named_steps') else best_model
            )
            if shap_results is not None:
                metrics['top_shap_features'] = shap_results.to_dict('records')
        except Exception as e:
            logging.warning(f"SHAP skipped: {str(e)}")

        # 6. Save outputs
        churn_model.save_model()
        churn_model.save_metrics(metrics)

        logging.info("Pipeline completed successfully")

    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()

