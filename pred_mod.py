import joblib
import numpy as n
import pandas as pd
from sklearn.ensemble._forest import RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor


def load_model(path) -> RandomForestRegressor:
    return joblib.load(path)

def prepare_data(id_df:pd.DataFrame, nodes_df:pd.DataFrame) -> pd.DataFrame:
    id_df = id_df.set_index('Id')
    id_df['running_time'] = pd.to_datetime(id_df['running_time'])
    id_df['day_time'] = id_df.running_time.dt.hour
    
    final_test_df = pd.concat([nodes_df.groupby('Id')['node_start'].count(), nodes_df.groupby('Id')['speed'].mean(),
           nodes_df.groupby('Id')['distance'].mean(), id_df['route_distance_km'], id_df['day_time']], axis=1)
    
    return final_test_df


def predict_RFR(test_df: pd.DataFrame, RFR:RandomForestRegressor) -> pd.Series:
       X = test_df.to_numpy()
       y_pred = RFR.predict(X)

       return pd.Series(y_pred, index = test_df.index)
