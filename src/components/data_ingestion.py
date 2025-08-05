import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass


@dataclass
class DataIngestionConfig: 
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")


class DataIngestion:

    def __init__(self, ingestion_config: DataIngestionConfig | None = None):
        self._ingestion_config = ingestion_config or DataIngestionConfig()

    def initiate_data_ingestion(self): 
        logging.info("Entered the data ingestion function")
        try: 
            df = pd.read_csv('notebook/data/stud.csv')
            logging.info(f"Got this data as a dataset: {df}")

            os.makedirs(os.path.dirname(self._ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self._ingestion_config.raw_data_path, index = False, header=True)
            logging.info("Trainn_test_split")

            train_set,test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self._ingestion_config.train_data_path,index=False, header=True)

            test_set.to_csv(self._ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")

            return (
                self._ingestion_config.train_data_path,
                self._ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys) 
        

if __name__ == "__main__": 
    odj = DataIngestion()
    one, two = odj.initiate_data_ingestion()