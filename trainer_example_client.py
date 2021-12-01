import grpc
import model_pb2
import model_pb2_grpc
import click
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import average_precision_score

from utils import task_type, standardise_dataset, create_loader


@click.command()
@click.option('--port', help='Enter a port')
def main(port):
    # open a gRPC channel
    channel = grpc.insecure_channel(port)

    # create a stub (client)
    stub = model_pb2_grpc.MoleculeTrainerStub(channel)

    # define required constants specific to the dataset
    smiles_col = "canonical_smiles_ap_nonstereo"    # name of the smiles column
    target_col = "wd_consensus_1"                   # name of the target column
    batch_size = 32
    gpu = 0                                         # nvidia-smi GPU id
    n_calls = 2                                     # number of iterations of Bayesian optimization
    n_random_starts = 1                             # number of random restarts of Bayesian optimization

    # define input configuration for the optimization procedure
    # run Bayesian otpimization wiith cross validation
    request_optimization = model_pb2.OptimizationConfig(
        data_filepath='./data/train.csv',
        smiles_col=smiles_col,
        target_col=target_col,
        batch_size=batch_size,
        seed=0,
        gpu=0,
        n_calls=n_calls,
        n_random_starts=n_random_starts,
    )

    response_optimization = stub.optimize(request_optimization)

    print('Optimized parameters for the model')
    print('Hidden channels: {}'.format(response_optimization.hidden_channels))
    print('Number of layers: {}'.format(response_optimization.num_layers))
    print('Number of heads: {}'.format(response_optimization.num_heads))
    print('Number of bases: {}'.format(response_optimization.num_bases))
    print('Learning rate: {}'.format(response_optimization.lr))

    # after the model is optimize you can train a single model for evaluation
    request_training = model_pb2.TrainingConfig(
        data_filepath='./data/train.csv',
        smiles_col=smiles_col,
        target_col=target_col,
        batch_size=batch_size,
        seed=0,
        gpu=0,
        hidden_channels=response_optimization.hidden_channels,
        num_layers=response_optimization.num_layers,
        num_heads=response_optimization.num_heads,
        num_bases=response_optimization.num_bases,
        lr=response_optimization.lr,
        name='evaluation'
    )
    response_training = stub.train(request_training)

    print('Directory of the trained model is')
    print(response_training.model_directory)

    """Evalaute the model on the test set"""
    #load the data
    test = pd.read_csv('./data/test.csv')
    predictions = []
    for smiles in tqdm(test[smiles_col]):
        request_prediction = model_pb2.Input(
            model_directory=response_training.model_directory,
            problem='classification',
            target_col=target_col,
            smiles=smiles
        )
        predictions.append(request_prediction.prediction)

    #Evaluate the model according to some metric
    average_precision = average_precision_score(test[target_col], predictions)
    print('Average precision on the test set: {}'.format(average_precision))





