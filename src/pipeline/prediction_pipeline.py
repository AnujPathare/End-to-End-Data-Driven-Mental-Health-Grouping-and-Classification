import sys

import pandas as pd

from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = "artifacts/model.pkl"
            preprocessor_path = "artifacts/preprocessor.pkl"

            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            data_scaled = preprocessor.transform(features)
            predictions = model.predict(data_scaled)
            return predictions
        
        except Exception as e:
            raise CustomException(e, sys)

# Class to map inputs given to HTML with backend
class CustomData:
    def __init__(self,
        Schizophrenia: float,
        Depressive: float,
        Anxiety: float,
        Bipolar: float,
        Eating: float,
    ):
        
        				
        self.Schizophrenia = Schizophrenia
        self.Depressive = Depressive
        self.Anxiety = Anxiety
        self.Bipolar = Bipolar
        self.Eating = Eating

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                "Schizophrenia": [self.Schizophrenia],
                "Depressive": [self.Depressive],
                "Anxiety": [self.Anxiety],
                "Bipolar": [self.Bipolar],
                "Eating": [self.Eating]
            }

            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            raise CustomException(e, sys)