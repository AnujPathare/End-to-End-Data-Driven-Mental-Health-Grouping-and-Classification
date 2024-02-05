import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

# Configuration settings: parameters such as source location, file formats, connection details, etc# 
@dataclass
class DataIngestionConfig:
    raw_data_path: str=os.path.join('artifacts', 'data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Initiated Data Ingestion")
        try:
            data = pd.read_csv('notebook/data/mental-illnesses.csv')
            df = data.drop(columns=['Entity', 'Code', 'Year'], axis=1)
            df.rename(columns={'Schizophrenia disorders (share of population) - Sex: Both - Age: Age-standardized':'Schizophrenia',
                   'Depressive disorders (share of population) - Sex: Both - Age: Age-standardized':'Depressive',
                  'Anxiety disorders (share of population) - Sex: Both - Age: Age-standardized':'Anxiety',
                  'Bipolar disorders (share of population) - Sex: Both - Age: Age-standardized':'Bipolar',
                  'Eating disorders (share of population) - Sex: Both - Age: Age-standardized':'Eating'},
                    inplace=True)
            logging.info("Read the dataset as dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok = True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Data Ingestion completed")

            return (
                self.ingestion_config.raw_data_path
                )

        except Exception as e:
            raise CustomException(e, sys)