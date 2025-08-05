import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_model

@dataclass
class ModelTrainerConfig: 
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig | None = None):
        self._model_trainer_config = model_trainer_config or ModelTrainerConfig()
    
    def initiate_model_trainer(self, train, test, preprocessor = None): 
        try: 
            logging.info("Splitting train and test data")
            X_train, y_train, X_test, y_test = (
                train[:,:-1],
                train[:,-1],
                test[:,:-1],
                test[:,-1]
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "GradientBoosting": GradientBoostingRegressor(),
                "LinearRegression": LinearRegression(),
                "K-Neighbors Classifier": KNeighborsRegressor(),
                "XGBRegressor": XGBRegressor(),
                "Catboosting Classifier": CatBoostRegressor(),
                "Adaboost Classifier": AdaBoostRegressor()
            }

            model_report:dict = evaluate_model(x_train=X_train, y_train=y_train, x_test = X_test, y_test = y_test, models=models)

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("Models are too bad")
            
            save_object(
                self._model_trainer_config.trained_model_file_path,
                best_model
            )

            predicted = best_model.predict(X_test)

            r2 = r2_score(y_test, predicted)
            return r2
        except Exception as e:
            raise CustomException(e, sys)


