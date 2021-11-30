from trainer import optimize, train


def main():
    hidden, layers, heads, bases, lr = optimize(
        data_filename="./data/train.csv",
        smiles_col="canonical_smiles_ap_nonstereo",
        target_col="wd_consensus_1",
        batch_size=32,
        seed=0,
        gpu=0,
        n_calls=2,
        n_random_starts=1,
    )

    train(
        data_filename="./data/train.csv",
        smiles_col="canonical_smiles_ap_nonstereo",
        target_col="wd_consensus_1",
        batch_size=32,
        seed=0,
        gpu=0,
        hidden_channels=hidden,
        num_layers=layers,
        num_heads=heads,
        num_bases=bases,
        lr=lr,
        name="testing"
    )

    #Add inference


    #Try regression


    #Add production training



if __name__ == '__main__':
    main()
