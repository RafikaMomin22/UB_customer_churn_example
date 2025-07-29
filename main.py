from churn_model.config import logging
from churn_model.prep.datasource import DataSource
from churn_model.prep.dataprep import DataPrep
from churn_model.model.model import ChurnModel
import pandas as pd


def main():
    logging.info("Starting churn prediction pipeline")

    # 1. Load data
    data_source = DataSource()
    orig_df = data_source.load_data()

    # 2. Preprocess and split
    data_prep = DataPrep(orig_df)
    x_train, x_test, y_train, y_test = data_prep.preprocess_data()

    # 3. Train models with CV
    churn_model = ChurnModel()
    best_model = churn_model.train_with_cv(x_train, y_train, data_prep.preprocessor)  # For feature names

    # 4. Evaluate on test set
    metrics = churn_model.evaluate_model(x_test, y_test)

    # 5. SHAP analysis (optional)
    try:
        x_test_df = pd.DataFrame(x_test, columns=data_prep.get_feature_names())
        shap_results = churn_model.analyze_shap(
            x_test_df,
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


if __name__ == "__main__":
    main()
