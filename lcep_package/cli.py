"""Console script for lcep-package."""
import os
import sys
from dataclasses import dataclass

import click
import numpy as np
import xgboost as xgb

from rich import traceback, print

WD = os.path.dirname(__file__)


@click.command()
@click.option('-i', '--input', type=str, help='Path to data file to predict.')
def main(input: str):
    """Console script for lcep-package."""
    print(r"""[bold blue]
    ██       ██████ ███████ ██████ 
    ██      ██      ██      ██   ██ 
    ██      ██      █████   ██████  
    ██      ██      ██      ██     
    ███████  ██████ ███████ ██ 
                               
        """)

    print('[bold blue]Run [green]cookietemple --help [blue]for an overview of all commands\n')

    model = get_xgboost_model(f'{WD}/models/model_28.08.2020_v1.xgb')
    data_to_predict = read_data_to_predict(input)
    predictions = np.round(model.predict(data_to_predict.DM))
    print(predictions)


@dataclass
class Dataset:
    X: np.ndarray
    y: list
    DM: xgb.DMatrix
    gene_names: list
    sample_names: list


def read_data_to_predict(path_to_data_to_predict: str) -> Dataset:
    """
    Parses the data to predict and returns a full Dataset include the DMatrix
    """
    X = []
    y = []
    gene_names = []
    sample_names = []
    with open(path_to_data_to_predict, "r") as file:
        all_runs_info = next(file).split("\n")[0].split("\t")[2:]
        for run_info in all_runs_info:
            split_info = run_info.split("_")
            y.append(int(split_info[0]))
            sample_names.append(split_info[1])
        for line in file:
            split = line.split("\n")[0].split("\t")
            X.append([float(x) for x in split[2:]])
            gene_names.append(split[:2])

    X = [list(i) for i in zip(*X)]

    X_np = np.array(X)
    DM = xgb.DMatrix(X_np, label=y)

    return Dataset(X_np, y, DM, gene_names, sample_names)


def get_xgboost_model(path_to_xgboost_model: str):
    """
    Fetches the model of choice and creates a booster from it.
    :param path_to_xgboost_model: Path to the xgboost model1
    """
    model = xgb.Booster()
    model.load_model(os.path.abspath(path_to_xgboost_model))

    return model


if __name__ == "__main__":
    traceback.install()
    sys.exit(main())  # pragma: no cover
