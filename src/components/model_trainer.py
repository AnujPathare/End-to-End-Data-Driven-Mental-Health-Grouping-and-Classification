import os
import sys
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
# from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score

from src.logger import logging
from src.exception import CustomException

from src.utils import save_object, evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            train_array = train_array.values
            test_array = test_array.values
            logging.info("Splitting Train and Test data")

            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "Logistic Regression": LogisticRegression(),
                "Support Vector Classifier": SVC(),
                "Gaussian NaiveBayes Classifier": GaussianNB(),
                "K Neighbors Classifier": KNeighborsClassifier(),
                "DecisionTree Classifier": DecisionTreeClassifier(),
                "RandomForest Classifier": RandomForestClassifier(),
                "AdaBoost Classifier": AdaBoostClassifier(),
                # "XGB Classifier": XGBClassifier(),
            }

            # params = {
            #     'Logistic Regression': {'penalty':['l2', 'elasticnet', None],'solver':['newton-cg','newton-cholesky','sag','saga']},
            #     'Support Vector Classifier': {'C':[0.01,0.4,0.7,1], 'kernel':['linear','poly','rbf','sigmoid'], 'gamma':['scale','auto']},
            #     'Gaussian NaiveBayes Classifier': {'var_smoothing': np.logspace(0,-9, num=25)},
            #     'K Neighbors Classifier': {'n_neighbors':[1,2,3,4,5,6,7],'weights':['uniform', 'distance'],'algorithm':['auto', 'ball_tree', 'kd_tree']},
            #     'DecisionTree Classifier': {'criterion': ['gini', 'entropy'],'max_depth': [3,4,5,6],'min_samples_split': [2,4,6,8,10]},
            #     'RandomForest Classifier': {'n_estimators':[10,25,50,100],'criterion':['gini', 'entropy', 'log_loss'],'max_depth':[3,4,5,6]},
            #     'AdaBoost Classifier': {'n_estimators':[10,25,50,100], 'learning_rate':[0.01,0.1,0.5,1], 'algorithm':['SAMME','SAMME.R']},
            #     # 'XGB Classifier': {'learning_rate': [0.01, 0.1, 0.2],'max_depth': [3,4,5,6],'n_estimators': [10,25,50,100]},
            # }

            logging.info("Starting Model Training")
            
            # model_report:dict = evaluate_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, params=params)
            model_report:dict = evaluate_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models)
            
            # Get best model based on recall score
            best_model_accuracy = max(sorted(model_report.values()))

            # Get best model's name
            # best_model_name = 'Gaussian NaiveBayes Classifier'
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_accuracy)
            ]
            best_model = models[best_model_name]

            logging.info("Model Training completed")
            
            if best_model_accuracy < 0.6:
                logging.info("No best model found")
            
            else:
                save_object(
                    file_path=self.model_trainer_config.trained_model_file_path,
                    obj=best_model
                )

                logging.info(f"Saving the Model: {best_model_name}")

                y_pred = best_model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                logging.info(f"Model Accuracy: {accuracy}")
                
                return accuracy

        except Exception as e:
            raise CustomException(e, sys)