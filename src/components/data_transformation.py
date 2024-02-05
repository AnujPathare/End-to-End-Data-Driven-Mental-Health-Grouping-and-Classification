import os
import sys
from dataclasses import dataclass

import pandas as pd
from sklearn.cluster import SpectralClustering
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from imblearn.combine import SMOTEENN

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    train_data_path = os.path.join('artifacts', 'train.csv')
    test_data_path = os.path.join('artifacts', 'test.csv')
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    # Function to get all pickle files
    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation
        '''
        try:
            features = ['Schizophrenia', 'Depressive', 'Anxiety', 'Bipolar', 'Eating']

            pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
            ])

            logging.info(f"Features: {features}")
 
            preprocessor = ColumnTransformer(
                transformers=[
                    ('pipeline', pipeline, features),
            ])
            
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
        
    
    def get_classification_data(self, raw_data):

        raw_data = raw_data

        spectral = SpectralClustering(n_clusters=2, assign_labels='discretize')
        spectral.fit(raw_data)

        cluster_df = pd.concat([raw_data, pd.DataFrame(data=spectral.labels_, columns=['Cluster'])], axis=1)

        return cluster_df


    def initiate_data_transformation(self, raw_data_path):

        try:
            data = pd.read_csv(raw_data_path)
            logging.info("Completed reading original data")

            logging.info("Obtaining classification data")
            df = self.get_classification_data(data)

            X = df.drop(columns=['Cluster'], axis=1)
            y = df['Cluster']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object()

            logging.info("Applying preprocessor object on training and testing dataframe")
            X_train_transformed = preprocessing_obj.fit_transform(X_train)
            X_test_transformed = preprocessing_obj.transform(X_test)


            logging.info("Resampling imbalanced dataset")
            smenn = SMOTEENN()
            X_resampled, y_resampled = smenn.fit_resample(X_train_transformed, y_train)

            features = ['Schizophrenia', 'Depressive', 'Anxiety', 'Bipolar', 'Eating']
            train_set = pd.concat([pd.DataFrame(X_resampled, columns=features), pd.DataFrame(y_resampled, columns=['Cluster'])], axis=1)
            y_test = y_test.reset_index(drop=True)
            test_set = pd.concat([pd.DataFrame(X_test_transformed, columns=features), y_test], axis=1)

            train_set.to_csv(self.data_transformation_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.data_transformation_config.test_data_path,index=False,header=True)

            logging.info("Saved train and test set")

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            logging.info("Saved preprocessing object")

            return (
                train_set,
                test_set,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)
        
