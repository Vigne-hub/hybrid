import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from models import MLPModule
from pre_processing import FoldingMLPData, FoldingMLPDataMFPT


def main(csv='mlp_dataset_simple_nbeads=30_withmfpt_2.csv'):
    # Assuming the correct database file path
    mlp_dataset = FoldingMLPDataMFPT()
    print(mlp_dataset.mlp_data.head())
    mlp_dataset.write_mlp_dataset_to_csv(csv)

    train_loader, val_loader, features, targets, scaler = (

        mlp_dataset.get_datasets(csv,
                                 query=None,
                                 val_size=0.2,
                                 batch_size=1,
                                 seed=10)
    )

    # Initialize the module
    model = MLPModule(features, targets, hidden_dims=[2 * len(features), 4], scaler=scaler)

    # Initialize the trainer
    trainer = L.Trainer(deterministic=True, max_epochs=100,

                        callbacks=[EarlyStopping(monitor='val_loss', patience=5, verbose=True)
                                   ])
    # Train the model
    trainer.fit(model, train_loader, val_loader)

    # Test the model
    trainer.test(model, val_loader)


def post_process_checkpoint(checkpoint_path, version=4, csv="mlp_dataset_simple_nbeads=30_withmfpt_2.csv"):
    train_loader, val_loader, num_features, num_targets, scaler = get_datasets(csv)

    model = MLPModule.load_from_checkpoint(checkpoint_path=checkpoint_path, num_features=num_features,
                                           num_outputs=num_targets, scaler=scaler)

    # Assuming the correct database file path
    mlp_dataset = FoldingMLPDataMFPT()
    print(mlp_dataset.mlp_data.head())
    mlp_dataset.write_mlp_dataset_to_csv(csv)

    train_loader, val_loader, features, targets, scaler = (

        mlp_dataset.get_datasets(csv,
                                 query='nbeads == 30',
                                 val_size=0.2,
                                 batch_size=1,
                                 seed=42)
    )

    # Initialize the module
    model = MLPModule.load_from_checkpoint(features=features, targets=targets,
                                           checkpoint_path=checkpoint_path, hidden_dims=[2 * len(features), 4],
                                           scaler=scaler)

    # Initialize the trainer
    trainer = L.Trainer(deterministic=True, max_epochs=5,

                        callbacks=[EarlyStopping(monitor='val_loss', patience=5, verbose=True)])

    # Test the model
    trainer.test(model, val_loader)


if __name__ == '__main__':
    main(csv="mlp_dataset_simple_nbeads=30_withmfpt_2.csv")
    # post_process("/cptg/u4/vrajesh/Documents/Margarita_project/hybridmc_ML/MLP/lightning_logs/version_27/checkpoints/epoch=41-step=118104.ckpt",
    #   version=27, csv="mlp_dataset_simple_nbeads=30_withmfpt_2.csv" )
    print('Done')
