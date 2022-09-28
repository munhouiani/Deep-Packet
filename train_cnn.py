import click

from ml.utils import train_application_classification_cnn_model, train_traffic_classification_cnn_model


@click.command()
@click.option('-d', '--data_path', help='training data dir path containing parquet files', required=True)
@click.option('-m', '--model_path', help='output model path', required=True)
@click.option('-t', '--task', help='classification task. Option: "app" or "traffic"', required=True)
def main(data_path, model_path, task):
    if task == 'app':
        train_application_classification_cnn_model(data_path, model_path)
    elif task == 'traffic':
        train_traffic_classification_cnn_model(data_path, model_path)
    else:
        exit('Not Support')


if __name__ == '__main__':
    main()
