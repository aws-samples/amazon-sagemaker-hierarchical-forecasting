import os
# os.system('pip install pandas')
# os.system('pip install scikit-hts')
# os.system('pip install plotly -q')
# os.system('pip install scikit-hts[prophet]')
# os.system('pip install scikit-hts[auto-arima]')

# os.system('conda install -c conda-forge fbprophet --yes')
import pandas as pd
import pathlib
import numpy as np
import argparse
import json
import boto3
from hts import HTSRegressor
import joblib
import ast
import plotly.graph_objects as go
from dataset_prep import prepare_data
import sklearn
from sklearn.metrics import mean_squared_error


import warnings
warnings.filterwarnings("ignore")

import logging
logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)

s3 = boto3.client("s3")

def train(bucket, seasonality_mode, algo, daily_seasonality, changepoint_prior_scale, revision_method):
    print('**************** Training Script ***********************')
    # create train dataset
    df = pd.read_csv(filepath_or_buffer=os.environ['SM_CHANNEL_TRAIN'] + "/train.csv", header=0, index_col=0)
    hierarchy, data, region_states = prepare_data(df)
    regions = df["region"].unique().tolist()
    # create test dataset
    df_test = pd.read_csv(filepath_or_buffer=os.environ['SM_CHANNEL_TEST'] + "/test.csv", header=0, index_col=0)
    test_hierarchy, test_df, region_states = prepare_data(df_test)
    print("************** Create Root Edges *********************")
    print(hierarchy)
    print('*************** Data Type for Hierarchy *************', type(hierarchy))
    # determine estimators##################################
    if algo == "Prophet":
        print('************** Started Training Prophet Model ****************')
        estimator = HTSRegressor(model='prophet', 
                                 revision_method=revision_method, 
                                 n_jobs=4, 
                                 daily_seasonality=daily_seasonality, 
                                 changepoint_prior_scale = changepoint_prior_scale,
                                 seasonality_mode=seasonality_mode,
                                )
        # train the model
        print("************** Calling fit method ************************")
        model = estimator.fit(data, hierarchy)
        print("Prophet training is complete SUCCESS")
        
        # evaluate the model on test data
        evaluate(model, test_df, regions, region_states)
    
    ###################################################
 
    mainpref = "scikit-hts/models/"
    prefix = mainpref + "/"
    print('************************ Saving Model *************************')
    joblib.dump(estimator, os.path.join(os.environ['SM_MODEL_DIR'], "model.joblib"))
    print('************************ Model Saved Successfully *************************')

    return model

def evaluate(model, test_df, regions, region_states):
    print("******* Making Predictions for 90days in future *********")
    predictions = model.predict(steps_ahead=90)
    test_preds = predictions.query('index > "2009-04-29"').copy()
    print('******************************', test_preds.shape)
    print('******************************', test_preds.head())
    ## root level metrics
    print('************ Root Level Metrics ************')
    total_mse = mean_squared_error(test_df['total'], test_preds['total'])
    print('Total: MSE: {}\n'.format(total_mse))
    
    print('************ Region Level Metrics ************')
    for region in regions:
            mse = mean_squared_error(test_df[region], test_preds[region])
            print('{}: MSE: {}\n'.format(region,mse))
        
    print('************ State Level Metrics ************')
    for state in region_states:
            mse = mean_squared_error(test_df[state], test_preds[state])
            print('{}:MSE: {}\n'.format(state,mse))

        
def model_fn(model_dir):
    predictor = joblib.load(os.path.join(model_dir, "model.joblib"))
    return predictor

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket', type=str, default="")
    parser.add_argument('--seasonality_mode', type=str, default="additive")
    parser.add_argument('--algo', type=str, default="Prophet")
    parser.add_argument('--daily_seasonality', type=bool, default=True)
    parser.add_argument('--changepoint_prior_scale', type=float, default=0.01)
    parser.add_argument('--revision_method', type=str, default='OLS')
#     parser.add_argument('--hybridize', type=bool, default=True)
#     parser.add_argument('--num_batches_per_epoch', type=int, default=10)    
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    train(args.bucket, args.seasonality_mode, args.algo, args.daily_seasonality, args.changepoint_prior_scale, args.revision_method)
  