from utils import standardise_dataset, smiles2graph, task_type, cross_val, create_loader
from skopt.space import Categorical, Integer, Real
from skopt.utils import use_named_args
from lightning_trainer import Conf, EGConvNet
import pytorch_lightning as pl
import numpy as np
from skopt import gp_minimize
from sklearn.model_selection import train_test_split
from pathlib import Path
from torch import Tensor
from torch.utils.data import WeightedRandomSampler
from torch_geometric.data import DataLoader
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import pandas as pd
from torch import load, Tensor, long, zeros
from standardiser import standardise

root = Path(__file__).resolve().parents[0].absolute()


def optimize(data_filepath, smiles_col, target_col, batch_size, seed, gpu, n_calls, n_random_starts):
    """Training with 5-fold cross validation for molecular prediction tasks"""

    print('Starting optimization preprocessing')
    data = pd.read_csv(data_filepath)
    data = data.loc[~data[target_col].isna()]
    # if the class of interest is the majority class or not heavily disbalanced optimize AUC
    # else optimize average precision
    # if regression optimize ap

    problem = task_type(data, target_col)

    # standardize the data
    data = standardise_dataset(data, smiles_col)

    # define params to optimize
    dim_1 = Categorical([128, 256, 512, 1024, 2048], name='hidden_channels')
    dim_2 = Integer(1, 8, name='num_layers')
    dim_3 = Categorical([2, 4, 8, 16], name='num_heads')
    dim_4 = Integer(1, 8, name='num_bases')
    dim_5 = Real(1e-5, 1e-3, name='lr')
    dimensions = [dim_1, dim_2, dim_3, dim_4, dim_5]

    @use_named_args(dimensions=dimensions)
    def inverse_ap(hidden_channels, num_layers, num_heads, num_bases, lr):
        fold_results = []
        conf = Conf(
            batch_size=batch_size,
            reduce_lr=True,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            num_heads=num_heads,
            num_bases=num_bases,
            lr=lr,
            seed=seed,
        )

        for fold in cross_val(data, target_col, problem=problem, batch_size=batch_size, seed=seed):
            model = EGConvNet(
                problem,
                conf.to_hparams(),
                reduce_lr=conf.reduce_lr,
            )

            early_stop_callback = EarlyStopping(monitor='result',
                                                min_delta=0.00,
                                                mode=('min' if problem == 'regression' else 'max'),
                                                patience=10,
                                                verbose=False)

            print("Starting training")
            trainer = pl.Trainer(
                max_epochs=1,
                gpus=[gpu],  # [0]  # load from checkpoint instead of resume
                weights_summary='top',
                callbacks=[early_stop_callback],
                logger=False,
                deterministic=True,
                auto_lr_find=False,
                num_sanity_val_steps=0,
                checkpoint_callback=False,
            )

            train_loader, val_loader, test_loader = fold
            trainer.fit(model, train_loader, val_loader)
            results = trainer.test(model, test_loader)
            if problem == 'ap':
                results = round(results[0]['test_ap'], 3)
            elif problem == 'auc':
                results = round(results[0]['test_auc'], 3)
            else:
                results = round(results[0]['test_mse'], 3)

            fold_results.append(results)

        print('Average metric across folds: {}'.format(np.mean(fold_results)))
        print('\n')

        for i, result in enumerate(fold_results):
            print('Metric for fold {}= {}'.format(i, result))

        return 1 / np.mean(fold_results)  # return inverse because you want to maximize it

    print('Starting Bayesian optimization')
    res = gp_minimize(inverse_ap,  # minimize the inverse of average precision
                      dimensions=dimensions,  # hyperparams
                      acq_func="EI",  # the acquisition function
                      n_calls=n_calls,  # the number of evaluations of f
                      n_random_starts=n_random_starts,  # the number of random initialization points
                      random_state=seed)  # the random seed

    print('Value of the minimum: {}'.format(res.fun))
    print('Res space: {}'.format(res.x))
    print('\n')

    results_path = Path(root / 'logs')

    if not results_path.exists():
        results_path.mkdir(exist_ok=True, parents=True)
        with open(results_path / "bayes_opt.txt", "w") as file:
            file.write("Bayes opt - EGConv")
            file.write("\n")

    with open("./logs/bayes_opt.txt", "a") as file:
        print('Target label: {}'.format(target_col), file=file)
        print('Hidden: {}'.format(res.x[0]), file=file)
        print('Layers: {}'.format(res.x[1]), file=file)
        print('Heads: {}'.format(res.x[2]), file=file)
        print('Bases: {}'.format(res.x[3]), file=file)
        print('Learning rate: {}'.format(res.x[4], file=file))
        print('Res space: {}'.format(res.space), file=file)
        file.write("\n")
        file.write("\n")

    return res.x[0], res.x[1], res.x[2], res.x[3], res.x[4]


def train(data_filepath, smiles_col, target_col, batch_size, seed, gpu,
          hidden_channels, num_layers, num_heads, num_bases, lr,
          name):
    """Train a model"""

    print('Starting training')
    data = pd.read_csv(data_filepath)
    data = data.loc[~data[target_col].isna()]

    # if the class of interest is the majority class or not heavily disbalanced optimize AUC
    # else optimize average precision
    # if regression optimize ap
    problem = task_type(data, target_col)

    # standardize the data
    data = standardise_dataset(data, smiles_col)

    if problem != 'regression':
        train, val = train_test_split(
            data,
            test_size=0.15,
            stratify=data[target_col],
            shuffle=True,
            random_state=seed
        )
    else:
        train, val = train_test_split(
            data,
            test_size=0.15,
            shuffle=True,
            random_state=seed
        )

    train_data_list = []
    for index, row in train.iterrows():
        train_data_list.append(smiles2graph(row, target_col))

    val_data_list = []
    for index, row in val.iterrows():
        val_data_list.append(smiles2graph(row, target_col))
    val_loader = DataLoader(val_data_list, num_workers=0, batch_size=batch_size)

    # if we are doing classification use weighted sampling for the minority class
    if problem != 'regression':
        minority = train[target_col].value_counts()[1]
        majority = train[target_col].value_counts()[0]
        class_sample_count = [majority, minority]
        weights = 1 / Tensor(class_sample_count)
        samples_weights = weights[train[target_col].values]
        sampler = WeightedRandomSampler(samples_weights,
                                        num_samples=len(samples_weights),
                                        replacement=True)
        train_loader = DataLoader(train_data_list, num_workers=0, batch_size=batch_size,
                                  sampler=sampler, drop_last=True)
    else:
        train_loader = DataLoader(train_data_list, num_workers=0, batch_size=batch_size, drop_last=True)

    conf = Conf(
        batch_size=batch_size,
        reduce_lr=True,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        num_heads=num_heads,
        num_bases=num_bases,
        lr=lr,
        seed=seed,
    )
    model = EGConvNet(
        problem,
        conf.to_hparams(),
        reduce_lr=conf.reduce_lr,
    )

    early_stop_callback = EarlyStopping(monitor='result',
                                        min_delta=0.00,
                                        mode=('min' if problem == 'regression' else 'max'),
                                        patience=10,
                                        verbose=False)

    logger = TensorBoardLogger(
        conf.save_dir,
        name='model',
        version='{}'.format(name),
    )

    model_checkpoint = ModelCheckpoint(
        dirpath=(logger.log_dir + '/checkpoint/'),
        monitor='result',
        mode=('min' if problem == 'regression' else 'max'),
        save_top_k=1,
    )

    print("Starting training")
    trainer = pl.Trainer(
        max_epochs=1,
        gpus=[gpu],  # [0]  # load from checkpoint instead of resume
        weights_summary='top',
        callbacks=[early_stop_callback, model_checkpoint],
        logger=False,
        deterministic=True,
        auto_lr_find=False,
        num_sanity_val_steps=0,
    )

    trainer.fit(model, train_loader, val_loader)

    return logger.log_dir


def predict(model_directory, problem, target_col, smiles):
    """Uses the trained model to make a prediction for an input molecule"""

    if problem == 'classification':
        problem = 'ap'
    else:
        problem = 'regression'

    # load the modal
    model_path = Path('{}/checkpoint/'.format(model_directory))
    files = model_path.glob(r'**/*.ckpt')
    files = [i for i in files][0]
    checkpoint = load(str(files))
    hparams = checkpoint['hyper_parameters']
    state_dict = checkpoint['state_dict']
    model = EGConvNet(problem=problem, hparams=hparams)
    model.load_state_dict(state_dict)
    model.eval()

    smiles = standardise.run(r'{}'.format(smiles))
    data = pd.DataFrame({'standardized_smiles': smiles}, index=[0]).iloc[0]
    data = smiles2graph(data, target_col)
    data.batch = zeros(data.num_nodes, dtype=long)
    output = model(data.x, data.edge_index, data.batch).detach().cpu().numpy()[0][0]
    output = round(((1 / (1 + np.exp(-output))) * 100), 2)

    return output
