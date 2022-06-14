import pandas as pd

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from prefect import flow, task
from prefect.task_runners import SequentialTaskRunner

from datetime import datetime
from dateutil.relativedelta import relativedelta

import pickle

@task
def read_data(path):
    df = pd.read_parquet(path)
    return df

@task
def prepare_features(df, categorical, train=True):
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    mean_duration = df.duration.mean()
    if train:
        print(f"The mean duration of training is {mean_duration}")
    else:
        print(f"The mean duration of validation is {mean_duration}")
    
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df

@task
def train_model(df, categorical):

    train_dicts = df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts) 
    y_train = df.duration.values

    print(f"The shape of X_train is {X_train.shape}")
    print(f"The DictVectorizer has {len(dv.feature_names_)} features")

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_train)
    mse = mean_squared_error(y_train, y_pred, squared=False)
    print(f"The MSE of training is: {mse}")
    return lr, dv

@task
def run_model(df, categorical, dv, lr):
    val_dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(val_dicts) 
    y_pred = lr.predict(X_val)
    y_val = df.duration.values

    mse = mean_squared_error(y_val, y_pred, squared=False)
    print(f"The MSE of validation is: {mse}")
    return
def get_paths(date=None):
    path_part="./data/fhv_tripdata_"
    if date == None:
        today = datetime.today()
    else:
        today = datetime.strptime(date, '%Y-%m-%d')
    train_path_td = today - relativedelta(months=2)
    val_path_td = today - relativedelta(months=1)
    train_datepart = train_path_td.strftime("%Y-%m")
    val_datepart = val_path_td.strftime("%Y-%m")
    return f"{path_part}{train_datepart}.parquet",f"{path_part}{val_datepart}.parquet"

@flow(task_runner=SequentialTaskRunner())
def main(date=None):
    train_path, val_path = get_paths(date)
    categorical = ['PUlocationID', 'DOlocationID']

    df_train = read_data(train_path)

    df_train_processed = prepare_features(df_train, categorical).result()

    df_val = read_data(val_path)

    df_val_processed = prepare_features(df_val, categorical, False).result()

    # train the model
    lr, dv = train_model(df_train_processed, categorical)
    
    if date == None:
        date = train_path[20:len(train_path)-8]
    
    with open(f"model-{date}.bin", "wb") as f_out:
            pickle.dump(lr, f_out)
    with open(f"dv-{date}.bin", "wb") as f_out:
            pickle.dump(dv, f_out)

    run_model(df_val_processed, categorical, dv, lr)

main(date="2021-08-15")

'''
from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import IntervalSchedule
from prefect.flow_runners import SubprocessFlowRunner
from datetime import timedelta

DeploymentSpec(
    flow=main_flow,
    name="homework",
    # schedule=IntervalSchedule(interval=timedelta(months=1)),
    flow_runner=SubprocessFlowRunner(),
    tags=["ml"],
)
'''
