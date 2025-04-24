import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from catboost import CatBoostClassifier


from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model


@dataclass
class DataTrainingConfig:
    trained_model_path = os.path.join("artifact", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = DataTrainingConfig()

    def initiate_model_training(self, train_array, test_array):
        try:
            logging.info("Splitting train and test input data")
            x_train, y_train, x_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Logistic Regression": LogisticRegression(),
                "K-Nearest Neighbors": KNeighborsClassifier(),
                "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
                "AdaBoost": AdaBoostClassifier(),
                "CatBoost": CatBoostClassifier(verbose=0)  # Disable training logs
            }

            params = {
                
            }

            # Fit and evaluate all models
            model_report, fitted_models = evaluate_model(x_train, y_train, x_test, y_test, models, params)

            best_model_score = max(model_report.values())
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = fitted_models[best_model_name]

            if best_model_score < 0.8:
                raise CustomException("No suitable model found with acceptable accuracy.")

            logging.info(f"Best model found: {best_model_name} with accuracy score: {best_model_score}")

            save_object(
                file_path=self.model_trainer_config.trained_model_path,
                obj=best_model
            )

            predicted = best_model.predict(x_test)
            accuracy = accuracy_score(y_test, predicted)
            logging.info(f"Final Model Accuracy Score: {accuracy}")

            return best_model, accuracy
        except Exception as e:
            logging.error(f"Error in model training: {str(e)}")
            raise CustomException(e, sys)