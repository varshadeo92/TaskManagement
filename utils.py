import os
import sys
import dill
import numpy as np
import pandas as pd
from src.exception import CustomException
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from src.logger import logging

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_model(x_train, y_train, x_test, y_test, models, params):
    try:
        report = {}
        fitted_models = {}

        for model_name, model in models.items():
            logging.info(f"Training model: {model_name}")
            if model_name in params and params[model_name]:  # Use GridSearchCV if params are available
                logging.info(f"Using GridSearchCV for {model_name}")
                param_grid = params[model_name]
                gs = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, verbose=1)
                gs.fit(x_train, y_train)
                model = gs.best_estimator_
                logging.info(f"Best params for {model_name}: {gs.best_params_}")
            else:
                model.fit(x_train, y_train)

            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)

            train_acc = accuracy_score(y_train, y_train_pred)
            test_acc = accuracy_score(y_test, y_test_pred)

            logging.info(f"{model_name} - Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}")

            report[model_name] = test_acc
            fitted_models[model_name] = model

        return report, fitted_models
    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path,"rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)
