import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object # To load our pickle file (model , preprocessor)

class PredictPipeline:
    def __init__(self):
        pass

    # This function will predict
    def predict(self,features):
        try:
            model_path = os.path.join("artifacts","model.pkl")
            preprocessor_path = os.path.join("artifacts","preprocessor.pkl")
            print("Before Loading")
            model = load_object(file_path= model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            print("After loading")
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)


## Custom data class will be responsible in mapping all the inputs that we give in html to the backend 
class CustomData:
    def __init__(self,
            Gender:str,Age:int,Height:int,Weight:int,Duration:int,Heart_Rate:int,Body_Temp:float):
        
        self.Gender = Gender
        self.Age = Age
        self.Height = Height
        self.Weight = Weight
        self.Duration = Duration
        self.Heart_Rate = Heart_Rate
        self.Body_Temp = Body_Temp

    # This function will return all the input in the form of data frame. 
    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Gender": [self.Gender],
                "Age": [self.Age],
                "Height": [self.Height],
                "Weight": [self.Weight],
                "Duration": [self.Duration],
                "Heart_Rate": [self.Heart_Rate],
                "Body_Temp": [self.Body_Temp],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)