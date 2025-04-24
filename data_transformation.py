import os
import sys
from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.exception import CustomException
from src.logger import logging
import pickle  # Importing for saving objects

def save_object(file_path, obj):
    """Function to save an object as a pickle file."""
    with open(file_path, "wb") as file:
        pickle.dump(obj, file)

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifact', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numeric_columns = ['Lodgeamt', 'ActualLossType', 'AGE', 'SEX', 'IPD_OPD', 'LOS']  # Fixed column names
            categorical_columns = [
                'Relation', 'Treatment_type'  # Fixed
            ]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy='median')),
                    ("scaler", StandardScaler())
                ]
            )
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy='most_frequent')),
                    ("one_hot_encoder", OneHotEncoder(handle_unknown='ignore', sparse_output=False)),  
                    ("scaler", StandardScaler())
                ]
            )

            logging.info("Numerical columns scaling pipeline created")
            logging.info("Categorical columns encoding pipeline created")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numeric_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            # Debugging: Print column names to verify correctness
            print("Columns in train_df:", train_df.columns)
            print("Columns in test_df:", test_df.columns)

            logging.info("Reading train and test data completed")
            logging.info("Obtaining preprocessing object")

            preprocessor_obj = self.get_data_transformer_object()
            target_column_name = "Opinion"  # Fixed column name

            # Ensure column names do not have extra spaces
            train_df.columns = train_df.columns.str.strip()
            test_df.columns = test_df.columns.str.strip()

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing on train and test data")

            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saving preprocessing object")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e, sys)
