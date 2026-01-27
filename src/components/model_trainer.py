import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import (
    RandomForestRegressor,
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    # Starting are model training
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            
            models = {
                "Linear Regression": LinearRegression(),
                "Random Forest": RandomForestRegressor(random_state=42,n_jobs=1),
                "Decision Tree Regressor": DecisionTreeRegressor()   
            }
            
            params={
            "Linear Regression" : {},

            "Random Forest" : {
            "n_estimators": [25, 50],
            "max_depth": [5, 8],
            "min_samples_split": [5, 10],
            "min_samples_leaf": [2, 5],
            "max_features": ["sqrt"]},

            "Decision Tree Regressor" : {
            "criterion": ["squared_error"],
            "max_depth": [5, 8, 10],
            "min_samples_split": [5, 10],
            "min_samples_leaf": [2, 5],
            "max_features": ["sqrt"]},
            }


            logging.info("Training model and finding model which will best fit")
            # Now we see which model is performing well
            # Model report we create a dictionary. Evaluate model is a function we create in utils
            model_report, trained_models = evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models,param=params)
            
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict
            # First we find the list of model score then we find the index of best model score. Then we find the model name and use that output.

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = trained_models[best_model_name]
            # eg if best_model_name = "Random Forest" then best_model = RandomForestRegressor()

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info("Model training completed")
            logging.info(f"Best found model on training and test dataset {best_model}")  

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            logging.info("model pickle file saved")
            predicted=best_model.predict(X_test)

            r_square = r2_score(y_test, predicted)
            return r_square
                        
        except Exception as e:
            raise CustomException(e,sys)