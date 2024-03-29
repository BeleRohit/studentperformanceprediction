import os
import sys
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer  # Used to create pipeline
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils import save_object



@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join ('artifacts', 'preprocessor_obj.pkl')
    model_file_path = os.path.join ('artifacts', 'model_trainer_config.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig ()

    def get_data_transformer(self):
        """
        This method is responsible for Data transformation.
        :return: preprocessor
        """
        try:
            num_features = ['reading_score', 'writing_score']
            cat_features = ['gender',
                            'race_ethnicity',
                            'parental_level_of_education',
                            'lunch',
                            'test_preparation_course']
            num_pipeline = Pipeline (
                steps = [("impute", SimpleImputer (strategy = "median"))
                    , ("scaler", StandardScaler (with_mean = False))
                         ])

            cat_pipeline = Pipeline (
                steps = [("impute", SimpleImputer (strategy = "most_frequent"))
                    , ("OneHotEncoder", OneHotEncoder ()),
                         ("scaler", StandardScaler (with_mean = False))
                         ]
            )
            logging.info ("Numerical Columns Standard Scaling Completed!")

            logging.info ("Categorical Column Encoding Completed!")
            preprocessor = ColumnTransformer (
                [
                    ("num", num_pipeline, num_features),
                    ("cat", cat_pipeline, cat_features)
                ]
            )
            return preprocessor

        except  Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info ("Data Reading Completed!")

            logging.info("Obtaining Preprocessing Object")

            preprocessor_object =self.get_data_transformer()

            target_col="math_score"
            numerical_cols=["writing_score","reading_score"]

            input_feature_train_df = train_df.drop (columns = [target_col], axis = 1)
            target_feature_train_df = train_df[target_col]
            input_feature_test_df = test_df.drop (columns = [target_col], axis = 1)
            target_feature_test_df = test_df[target_col]

            logging.info (
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr = preprocessor_object.fit_transform (input_feature_train_df)
            input_feature_test_arr = preprocessor_object.transform (input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array (target_feature_test_df)]

            logging.info (f"Saved preprocessing object.")

            save_object (

                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessor_object

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException (e, sys)