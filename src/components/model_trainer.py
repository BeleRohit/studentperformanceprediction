import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models
@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model_trainer_config.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_df,test_df):
        try:
            logging.info("Initiating model_trainer_config trainer")
            logging.info("Splitting the test and training data")
            X_train,y_train,X_test,y_test =(
                train_df[:,:-1],train_df[:,-1],
                test_df[:,:-1],test_df[:,-1]
            )
            models = {
                "Random Forest": RandomForestRegressor (),
                "Decision Tree": DecisionTreeRegressor (),
                "Gradient Boosting": GradientBoostingRegressor (),
                "Linear Regression": LinearRegression (),
                "XGBRegressor": XGBRegressor (),
                "CatBoosting Regressor": CatBoostRegressor (verbose = False),
                "AdaBoost Regressor": AdaBoostRegressor (),
            }
            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models = models)
            best_model_score=max(sorted(model_report.values()))

            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model=models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best Model Found for this dataset")

            logging.info("Best found model_trainer_config on both training and testing dataset")

            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )
            predicted=best_model.predict(X_test)

            r2_square=r2_score(y_test,predicted)

            return (
                best_model_name,
                r2_square
            )

        except Exception as e:
            raise CustomException(e,sys)

