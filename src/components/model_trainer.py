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
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_training(self, train_array, test_array):
        try:
            logging.info("Splitting train-test input data")
            x_train, y_train, x_test, y_test = (
                train_array[:,:-1], # Stores all column except the last column
                train_array[:,-1], # Stores the last column
                
                test_array[:,:-1], # Stores all column except the last column
                test_array[:,-1], # Stores the last column          
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            params = {
                "Decision Tree": {
                    "criterion": ["squared_error", "absolute_error", "poisson"],
                    "max_depth": [None, 5, 10],
                    "min_samples_split": [2, 5],
                    "min_samples_leaf": [1, 2]
                },

                "Random Forest": {
                    "n_estimators": [50, 100],
                    "max_depth": [None, 10],
                    "min_samples_split": [2, 5],
                    "min_samples_leaf": [1, 2]
                },

                "Gradient Boosting": {
                    "n_estimators": [50, 100],
                    "learning_rate": [0.05, 0.1],
                    "max_depth": [2, 3],
                    "subsample": [0.8, 1.0]
                },

                "Linear Regression": {},

                "XGBRegressor": {
                    "n_estimators": [50, 100],
                    "learning_rate": [0.05, 0.1],
                    "max_depth": [3, 5],
                    "subsample": [0.8, 1.0]
                },

                "CatBoosting Regressor": {
                    "iterations": [100, 200],
                    "depth": [4, 6],
                    "learning_rate": [0.05, 0.1]
                },

                "AdaBoost Regressor": {
                    "n_estimators": [50, 100],
                    "learning_rate": [0.05, 0.1]
                }
            }


            model_report:dict=evaluate_models(X_train=x_train,y_train=y_train,X_test=x_test,y_test=y_test,models=models,param=params)
            
            # To get best model Score from dict
            best_model_score = max(sorted(model_report.values()))

            # To get best model Name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")

            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(x_test)

            r2_square = r2_score(y_test, predicted)
            return r2_square

        except Exception as e:
            raise CustomException(e, sys)