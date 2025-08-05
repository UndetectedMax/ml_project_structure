import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path:str = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self, data_transformation_config: DataTransformationConfig | None = None):
        self._data_transformation_config = data_transformation_config or DataTransformationConfig() 

    def _get_data_transformer_object(self):
        '''
        
        '''
        try: 
            numerical_columns = ["writing_score","reading_score"]
            categorical_features = ["gender","race_ethnicity", "parental_level_of_education", "lunch", "test_preparation_course"]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ]
            )
            cat_pipeline = Pipeline(
                steps = [
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            logging.info("Both Pipelines are created")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_features)
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e, sys) 
        
    def initiate_data_trandfornation(self, train_path, test_path): 
        try: 
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Imported the train and test data")

            preprocessor_odj = self._get_data_transformer_object()
            target_column_name = "math_score"
            numerical_columns = ["writing_score","reading_score"]

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying the preprocessor object on train test")

            input_feature_train_arr = preprocessor_odj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_odj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
                
            ]
            logging.info("Save all preprocessing objects")

            save_object_obj = save_object(
                file_path = self._data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_odj
            )

            return (
                train_arr,
                test_arr,
                self._data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e, sys)