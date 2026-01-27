import os 
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from src.components.data_ingestion import DataIngestion
from src.components.data_ingestion import DataIngestionConfig
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainer
from src.components.model_trainer import ModelTrainerConfig




if __name__ == "__main__":
    try:
        data_ingestion = DataIngestion()
        logging.info("Data ingestion started")
        train_data, test_data = data_ingestion.initiate_data_ingestion()
        logging.info("Data ingestion completed")

        data_transformation = DataTransformation()
        logging.info("Data transformation started")
        train_arr, test_arr, _ = data_transformation.initiate_data_transformation(
            train_data, test_data
        )
        logging.info("Data transformation completed")

        logging.info("Model training started")
        model_trainer = ModelTrainer()
        result = model_trainer.initiate_model_trainer(train_arr, test_arr)
        logging.info(f"Model training completed: {result}")

    except Exception as e:
        raise CustomException(e,sys)
            
