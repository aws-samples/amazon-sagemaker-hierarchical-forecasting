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

import warnings
warnings.filterwarnings("ignore")

import logging
logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)

s3 = boto3.client("s3")
# Aggregate data by region and Quantity
def get_region_columns(df, region):
    return [col for col in df.columns if region in col]

def prepare_data(df_raw):
    print('******************* Prepare Data **********************')
    product = df_raw[(df_raw['item'] == "Onion")]
    product["region_state"] = product.apply(lambda x: f"{x['region']}_{x['state']}", axis=1)
    region_states = product["region_state"].unique()
    grouped_sections = product.groupby(["region", "region_state"])
    edges_hierarchy = list(grouped_sections.groups.keys())
    # Now, we must not forget total that is our root node.
    second_level_nodes = product.region.unique()
    root_node = "total"
    root_edges = [(root_node, second_level_node) for second_level_node in second_level_nodes]
    root_edges += edges_hierarchy
    product_bottom_level = product.pivot(index="date", columns="region_state", values="quantity")
    regions = product["region"].unique().tolist()
    for region in regions:
        region_cols = get_region_columns(product_bottom_level, region)
        product_bottom_level[region] = product_bottom_level[region_cols].sum(axis=1)

    product_bottom_level["total"] = product_bottom_level[regions].sum(axis=1)
   
    # create hierarchy
    hierarchy = dict()

    for edge in root_edges:
        parent, children = edge[0], edge[1]
        hierarchy.get(parent)
        if not hierarchy.get(parent):
            hierarchy[parent] = [children]
        else:
            hierarchy[parent] += [children]
    
    product_bottom_level.index = pd.to_datetime(product_bottom_level.index)
    product_bottom_level = product_bottom_level.resample("D").sum()
    print('******************* End Prepare Data **********************')
    return hierarchy, product_bottom_level, region_states