from trainer import optimize, train
from lightning_trainer import EGConvNet
from utils import create_loader, standardise_dataset, task_type
from pathlib import Path
import pandas as pd
import torch
from pytorch_lightning import Trainer


def main():
    # define required parameters specific to the dataset
    smiles_col = "Drug"
    target_col = "Y"
    batch_size = 32

    # run Bayesian otpimization wiith cross validation
    hidden, layers, heads, bases, lr = optimize(
        data_filename="./data/bioavailability_ma.csv",
        smiles_col=smiles_col,
        target_col=target_col,
        batch_size=batch_size,
        seed=0,
        gpu=0,
        n_calls=2,
        n_random_starts=1,
    )

    # Train the model and evaluate on an outer test set
    name = 'evaluation'  # choose a name for the folder where the model is to be saved
    train(
        data_filename="./data/bioavailability_ma.csv",
        smiles_col=smiles_col,
        target_col=target_col,
        batch_size=batch_size,
        seed=0,
        gpu=0,
        hidden_channels=hidden,
        num_layers=layers,
        num_heads=heads,
        num_bases=bases,
        lr=lr,
        name=name
    )

    """Evalaute the model on the test set"""
    #load the data
    test = pd.read_csv('./data/bioavailability_ma.csv')
    problem = task_type(test, target_col)
    test = standardise_dataset(test, smiles_col)  # don't forget to standardise the test set smiles
    test_loader = create_loader(test, target_col, batch_size=batch_size)

    #load the modal
    withdrawn_model_path = Path('./model/{}/checkpoint/'.format(name))
    files = withdrawn_model_path.glob(r'**/*.ckpt')
    files = [i for i in files]
    checkpoint = torch.load(str(files[0]))
    hparams = checkpoint['hyper_parameters']
    state_dict = checkpoint['state_dict']
    model = EGConvNet(problem=problem, hparams=hparams)
    model.load_state_dict(state_dict)

    #initatie pl trainer so we can run test faster
    trainer = Trainer(logger=False)
    results = trainer.test(model, test_loader)
    print(results[0])

    """Deploying models in production
    * To deploy models in production you can use the training function on the combined training and test datasets"""

    # Inference on single samples can be done by calling model.forward() on the model
    model.eval()  # set this flag when doing inference, not required when running model.test()

    #define an input molecule
    input = pd.DataFrame(
        {'name': 'Azithromycin',
         'smiles': 'CCC1C(C(C(N(CC(CC(C(C(C(C(C(=O)O1)C)OC2CC(C(C(O2)C)O)(C)OC)C)OC3C(C(CC(O3)C)N(C)C)O)(C)O)C)C)C)O)(C)O'},
        index=[0]
    )

    #standardize the molecule
    input = standardise_dataset(input, smiles_col='smiles')

    #create loader
    input_loader = create_loader(input, smiles_col='smiles', target_col=None, batch_size=batch_size)

    #iterate through loader and make a prediction
    predictions = []
    for data in input_loader:
        output = model.forward(data)
        predictions.append(output)
    print(predictions)


if __name__ == '__main__':
    main()
