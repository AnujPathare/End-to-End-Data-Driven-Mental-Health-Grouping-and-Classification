from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


if __name__ == "__main__":

    data_ingestion = DataIngestion()
    raw_data = data_ingestion.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_set, test_set, _ = data_transformation.initiate_data_transformation(raw_data)

    model_trainer = ModelTrainer()
    model_trainer.initiate_model_trainer(train_set, test_set)