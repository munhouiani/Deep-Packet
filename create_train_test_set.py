import os
import sys
from pathlib import Path

import click
import psutil
from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import col, monotonically_increasing_id, lit, row_number, rand


def top_n_per_group(spark_df, groupby, topn):
    spark_df = spark_df.withColumn('rand', rand(seed=9876))
    window = Window.partitionBy(col(groupby)).orderBy(col('rand'))

    return (
        spark_df
            .select(col('*'), row_number().over(window).alias('row_number'))
            .where(col('row_number') <= topn)
            .drop('row_number', 'rand')
    )


def split_train_test(df, test_size, under_sampling_train=True):
    # add increasing id for df
    df = df.withColumn('id', monotonically_increasing_id())

    # stratified split
    fractions = (
        df
            .select('label')
            .distinct().
            withColumn('fraction', lit(test_size))
            .rdd
            .collectAsMap()
    )
    test_id = (
        df
            .sampleBy('label', fractions, seed=9876)
            .select('id')
            .withColumn('is_test', lit(True))
    )

    df = df.join(test_id, how='left', on='id')

    train_df = (
        df
            .filter(col('is_test').isNull())
            .select('feature', 'label')
    )
    test_df = (
        df.
            filter(col('is_test'))
            .select('feature', 'label')
    )

    # under sampling
    if under_sampling_train:
        # get label list with count of each label
        label_count_df = (
            train_df.
                groupby('label')
                .count()
                .toPandas()
        )

        # get min label count in train set for under sampling
        min_label_count = int(label_count_df['count'].min())

        train_df = top_n_per_group(train_df, 'label', min_label_count)

    return train_df, test_df


def save_parquet(df, path):
    output_path = path.absolute().as_uri()
    (
        df
            .write
            .mode('overwrite')
            .parquet(output_path)
    )


def save_train(df, path_dir):
    path = path_dir / 'train.parquet'
    save_parquet(df, path)


def save_test(df, path_dir):
    path = path_dir / 'test.parquet'
    save_parquet(df, path)


def create_train_test_for_task(df, label_col, test_size, under_sampling, data_dir_path):
    task_df = df.filter(col(label_col).isNotNull()).selectExpr('feature', f'{label_col} as label')
    print('splitting train test')
    train_df, test_df = split_train_test(task_df, test_size, under_sampling)
    print('splitting train test done')
    print('saving train')
    save_train(train_df, data_dir_path)
    print('saving train done')
    print('saving test')
    save_test(test_df, data_dir_path)
    print('saving test done')


def print_df_label_distribution(spark, path):
    print(path)
    print(
        spark
            .read
            .parquet(path.absolute().as_uri())
            .groupby('label').count().toPandas()
    )


@click.command()
@click.option('-s', '--source', help='path to the directory containing preprocessed files', required=True)
@click.option('-t', '--target',
              help='path to the directory for persisting train and test set for both app and traffic classification',
              required=True)
@click.option('--test_size', default=0.2, help='size of test size', type=float)
@click.option('--under_sampling', default=True, help='under sampling training data', type=bool)
def main(source, target, test_size, under_sampling):
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

    # read data
    df = spark.read.parquet(f'{source_data_dir_path.absolute().as_uri()}/*.parquet')

    # prepare data for application classification and traffic classification
    print('processing application classification dataset')
    create_train_test_for_task(df=df, label_col='app_label', test_size=test_size, under_sampling=under_sampling,
                               data_dir_path=application_data_dir_path)

    print('processing traffic classification dataset')
    create_train_test_for_task(df=df, label_col='traffic_label', test_size=test_size, under_sampling=under_sampling,
                               data_dir_path=traffic_data_dir_path)

    # stats
    print_df_label_distribution(spark, application_data_dir_path / 'train.parquet')
    print_df_label_distribution(spark, application_data_dir_path / 'test.parquet')
    print_df_label_distribution(spark, traffic_data_dir_path / 'train.parquet')
    print_df_label_distribution(spark, traffic_data_dir_path / 'test.parquet')


if __name__ == '__main__':
    main()
