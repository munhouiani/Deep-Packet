import os
import sys
from pathlib import Path

import psutil
from petastorm.codecs import CompressedNdarrayCodec, ScalarCodec
from petastorm.unischema import Unischema, UnischemaField
from pyspark.sql import SparkSession
from pyspark.sql.types import LongType

import numpy as np

source = '/home/munhou/DeepPacket/processed_data'
target = '/home/munhou/DeepPacket/www'
source_data_dir_path = Path(source)
target_data_dir_path = Path(target)

# prepare dir for dataset
application_data_dir_path = target_data_dir_path / 'application_classification'
traffic_data_dir_path = target_data_dir_path / 'traffic_classification'

# initialise local spark
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
memory_gb = psutil.virtual_memory().available // 1024 // 1024 // 1024
spark = (
    SparkSession
        .builder
        .master('local[*]')
        .config('spark.driver.memory', f'{memory_gb}g')
        .config('spark.driver.host', '127.0.0.1')
        .getOrCreate()
)

# prepare final schema
schema = Unischema(
    'data_schema', [
        UnischemaField('feature', np.float32, (1, 1500), CompressedNdarrayCodec(), False),
        UnischemaField('label', np.int32, (), ScalarCodec(LongType()), False),
    ]
)
# %%
# read data
df = spark.read.parquet(f'{source_data_dir_path.absolute().as_uri()}/*.parquet').drop('feature')
