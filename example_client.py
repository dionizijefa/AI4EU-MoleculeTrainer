from trainer import optimize, train
from lightning_trainer import EGConvNet
from utils import create_loader, standardise_dataset, task_type
from pathlib import Path
import pandas as pd


def main():
    # define required parameters specific to the dataset
    smiles_col = "canonical_smiles_ap_nonstereo"
    target_col = "wd_consensus_1"
    batch_size=32

    """
    hidden, layers, heads, bases, lr = optimize(
        data_filename="./data/train.csv",
        smiles_col=smiles_col,
        target_col=target_col,
        batch_size=batch_size,
        seed=0,
        gpu=0,
        n_calls=2,
        n_random_starts=1,
    )

    name = 'test'  # define that this is a model for evaluation on the test
    train(
        data_filename="./data/train.csv",
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
    """
    name="test"
    # Evaluate on the test set
    withdrawn_model_path = Path('./model/{}/checkpoint/'.format(name))
    files = withdrawn_model_path.glob(r'**/*.ckpt')
    files = [i for i in files]
    print(files)
    model = EGConvNet.load_from_checkpoint(checkpoint_path=files[0])
    model.eval()  # set this flag when doing inference, not required when running model.test

    test = pd.read_csv('./data/test.csv')
    problem = task_type(test, target_col)
    test = standardise_dataset(test, smiles_col)  # don't forget to standardise the test set smiles
    test_loader = create_loader(test, target_col, batch_size=batch_size)
    result = model.test(test_loader)
    print(result[0])

    # Try regression

    # Add production training


if __name__ == '__main__':
    main()
