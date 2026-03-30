import os
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from src.logger import get_logger
from src.custom_exception import CustomException

import mlflow
import mlflow.sklearn

logger = get_logger(__name__)


class ModelTraining:
    def __init__(self, processed_data_path="artifacts/processed"):
        self.processed_data_path = processed_data_path
        self.model_dir = "artifacts/models"
        os.makedirs(self.model_dir, exist_ok=True)
        logger.info("Model Training Initialization...")

    def load_data(self):
        try:
            self.x_train = joblib.load(os.path.join(self.processed_data_path, "x_train.pkl"))
            self.x_test = joblib.load(os.path.join(self.processed_data_path, "x_test.pkl"))
            self.y_train = joblib.load(os.path.join(self.processed_data_path, "y_train.pkl"))
            self.y_test = joblib.load(os.path.join(self.processed_data_path, "y_test.pkl"))
            logger.info("Data loaded for model....")

        except Exception as e:
            logger.error(f"Error while loading data for model: {e}")
            raise CustomException(f"Failed to load data for model: {e}")

    def train_model(self):
        try:
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=42,
            )
            self.model.fit(self.x_train, self.y_train)

            joblib.dump(self.model, os.path.join(self.model_dir, "model.pkl"))
            logger.info("Model trained and saved successfully...")

        except Exception as e:
            logger.error(f"Error while training model: {e}")
            raise CustomException(f"Failed to train model: {e}")

    def evaluate_model(self):
        try:
            y_pred = self.model.predict(self.x_test)

            y_unique = set(self.y_test)
            if len(y_unique) == 2:
                y_proba = self.model.predict_proba(self.x_test)[:, 1]
                roc_auc = roc_auc_score(self.y_test, y_proba)
                logger.info(f"ROC-AUC Score: {roc_auc}")
            else:
                y_proba = None
                logger.info("ROC-AUC skipped because target is not binary.")

            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, average="weighted")
            recall = recall_score(self.y_test, y_pred, average="weighted")
            f1 = f1_score(self.y_test, y_pred, average="weighted")
            mlflow.log_metric("accuracy",accuracy)
            mlflow.log_metric("Precision",precision)
            mlflow.log_metric("Recall_Score",recall)
            mlflow.log_metric("F1 Score",f1)

            logger.info(
                f"Accuracy: {accuracy}; Precision: {precision}; Recall: {recall}; F1: {f1}"
            )

            logger.info("Model evaluation done...")

        except Exception as e:
            logger.error(f"Error while evaluating model: {e}")
            raise CustomException(f"Failed to evaluate the model: {e}")

    def run(self):
        self.load_data()
        self.train_model()
        self.evaluate_model()


if __name__ == "__main__":
    with mlflow.start_run():
        trainer = ModelTraining()
        trainer.run()