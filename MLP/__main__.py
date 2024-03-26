import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from models import MLPModule
from pre_processing import FoldingMLPData, FoldingMLPDataMFPT
from sklearn.model_selection import train_test_split, KFold
from lightning.pytorch import seed_everything
import pandas as pd
import numpy as np
import csv
from functools import partial

import multiprocessing
from multiprocessing import Pool


def main(csv_name, write=1, target_columns=None, seed=42):
    seed_everything(seed, workers=True)
    # Assuming the correct database file path
    mlp_dataset = FoldingMLPDataMFPT()
    print(mlp_dataset.mlp_data.head())

    if write:
        mlp_dataset.write_mlp_dataset_to_csv(csv_name)

    if target_columns is not None:
        mlp_dataset.target_columns = [el for el in target_columns]

    train_loader, val_loader, features, targets, scaler = (

        mlp_dataset.get_datasets(csv_name,
                                 query=None,
                                 val_size=0.2,
                                 batch_size=1,
                                 seed=seed)
    )

    # Initialize the module
    i = len(features)
    model = MLPModule(features, targets, hidden_dims=[10, 6, 4], scaler=scaler, learning_rate=1e-2)

    # Initialize the trainer
    trainer = L.Trainer(deterministic=True, max_epochs=100,

                        callbacks=[EarlyStopping(monitor='val_loss', patience=5, verbose=True)
                                   ])
    # Train the model
    trainer.fit(model, train_loader, val_loader)

    # Test the model
    trainer.test(model, val_loader)


def main_sizes(csv_name, target_columns, seed=42, write=False, version=1):
    seed_everything(seed, workers=True)
    # Assuming the correct database file path
    mlp_dataset = FoldingMLPDataMFPT()
    print(mlp_dataset.mlp_data.head())

    if write:
        mlp_dataset.write_mlp_dataset_to_csv(csv_name)

    if target_columns is not None:
        mlp_dataset.target_columns = [el for el in target_columns]

    # Placeholder for aggregate model performance across folds
    test_losses = [("Training_set_size", 'Valdiation_set_size', 'test_loss')]

    # Fractions to test
    val_fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    for val_size in val_fractions:
        print(f"Training with {val_size} validation size.")

        train_loader, val_loader, features, targets, scaler = (

            mlp_dataset.get_datasets(csv_name,
                                     query='nbeads == 25',
                                     val_size=val_size,
                                     batch_size=1,
                                     seed=42,
                                     second_split=1)
        )

        # Initialize the model for this fold
        i = len(features)
        model = MLPModule(features, targets, hidden_dims=[10, 6, 4], scaler=scaler, learning_rate=1e-2)

        # Initialize the trainer
        trainer = L.Trainer(deterministic=True, max_epochs=100,

                            callbacks=[EarlyStopping(monitor='val_loss', patience=5, verbose=True)
                                       ])

        # Train the model on this fold's training data
        trainer.fit(model, train_loader, val_loader)

        # Optionally, evaluate the model on this fold's validation data
        trainer.test(model, val_loader)

        # Store or process this fold's performance for aggregation
        # This could be validation loss, accuracy, or any other metric of interest
        test_losses.append(
            (len(train_loader.dataset), len(val_loader.dataset), float(trainer.logged_metrics['test_loss'])))

    # write test losses to csv
    with open(f"size_test{version}.csv", mode='w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(test_losses)

    print(f"Average performance: {np.mean(np.array(test_losses)[:, 2])}")


def train_model(val_size, mlp_dataset, csv_name):
    print(f"Training with {val_size} validation size.")

    train_loader, val_loader, features, targets, scaler = (
        mlp_dataset.get_datasets(csv_name,
                                 query='nbeads == 25',
                                 val_size=val_size,
                                 batch_size=1,
                                 seed=42,
                                 second_split=1)
    )

    # Initialize the model for this fold
    model = MLPModule(features, targets, hidden_dims=[10, 6, 4], scaler=scaler, learning_rate=1e-2)

    # Initialize the trainer
    trainer = L.Trainer(deterministic=True, max_epochs=100,
                        callbacks=[EarlyStopping(monitor='val_loss', patience=5, verbose=True)])

    # Train the model
    trainer.fit(model, train_loader, val_loader)

    # Optionally, evaluate the model
    trainer.test(model, val_loader)

    # Return performance metrics
    return (len(train_loader.dataset), len(val_loader.dataset), float(trainer.logged_metrics['test_loss']))


def P_main(version, seed, target_columns):
    seed_everything(seed, workers=True)
    # Assuming the correct database file path
    mlp_dataset = FoldingMLPDataMFPT()
    print(mlp_dataset.mlp_data.head())

    if target_columns is not None:
        mlp_dataset.target_columns = [el for el in target_columns]

    process_func = partial(train_model,
                           csv_name="mlp_dataset_nbeads=25.csv",
                           mlp_dataset=mlp_dataset
                           )

    val_fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    with Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.map(process_func, val_fractions)

    # Process results
    average_test_loss = sum(result[2] for result in results) / len(results)
    print(f"Average test loss: {average_test_loss}")

    # write test losses to csv
    with open(f"size_test{version}.csv", mode='w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Training_set_size", 'Valdiation_set_size', 'test_loss'])
        writer.writerows(results)


if __name__ == '__main__':
    P_main(version=1, seed=42, target_columns=['s_bias_mean'])

    # main_sizes(csv_name="mlp_dataset_nbeads=25.csv", write=0, target_columns=['s_bias_mean'])

    # post_process("/cptg/u4/vrajesh/Documents/Margarita_project/hybridmc_ML/MLP/lightning_logs/version_27/checkpoints/epoch=41-step=118104.ckpt",
    #   version=27, csv="mlp_dataset_simple_nbeads=30_withmfpt_2.csv" )
    print('Done')
