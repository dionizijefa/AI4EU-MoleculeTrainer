from trainer import optimize, train

optimize(
    data_filename="data/train.csv",
    smiles_col="canonical_smiles_ap_nonstereo",
    target_col="wd_consensus_1",
    batch_size=32,
    seed=0,
    gpu=0
)
