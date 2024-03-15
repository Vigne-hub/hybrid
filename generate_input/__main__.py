import os
from pathlib import Path
from . import ConfigGeneratorDriver, SubmitConfigJob, MLDatabaseManager, MLDatabaseManagerParallel
import configparser


def run_generation(gen, submit, create_job_script, integrate_output):

    # everything is done assuming the working directory is where the module is located.
    # TODO: Generalize this before publishing.
    os.chdir(Path(__file__).parent)

    if not (gen or submit or create_job_script):
        if integrate_output:
            integrate_csv_wrapper(settings_config_file="settings.cfg")
        else:
            from sys import exit
            exit("Nothing to do. Please pass argument to indicate what to do.")

    if args.gen:
        # Generate some configurations
        config_generator = ConfigGeneratorDriver(settings_config_file="settings.cfg")
        config_generator.generate_configs()

    if submit or create_job_script:

        # read config settings
        config = configparser.ConfigParser()

        # submit job to slurm for these configs
        jobsubmitter = SubmitConfigJob(settings_config_file="settings.cfg")

        if args.submit:
            jobsubmitter.submit_job()

        if create_job_script:
            jobsubmitter.create_job_script()
            print(f"Shell script in {jobsubmitter.temp_script_path}")


def integrate_csv_wrapper(settings_config_file):
    """
    Function to integrate the various csv outputs from hybridmc like diff_sbias and avg_s_bias from the simulation
    output directory for that structure, to the configuration database file with it.

    :param settings_config_file: Path to the settings configuration files,
    setting_config_file should have the following parameters:

        :param db_path: Database path containing the structures for which the csv outputs are found and integrated with.
        :param search_path: PAth to recursively search for the csv_name

    """

    from pathlib import Path

    config = configparser.ConfigParser()
    config.read(settings_config_file)

    # path to the directory with all the generated structures with the catalog of them in a config.db file
    target_directory = Path(config.get('master_settings', 'target_directory', fallback="sim_configs"))
    # path to directory with hybridmc simulation result directories for each structure run
    out_dir = config.get('slurm_settings', 'out_dir', fallback="sample_train")

    # path to the .db file in target directory
    db_path = target_directory / config.get('master_settings', 'database_name', fallback="configurations.db")

    # open connection to the database
    manager = MLDatabaseManagerParallel(db_path)

    # integrate the diff_s_bias data. These outputs do not have headers so include these manually
    manager.integrate_csv_data(csv_name="diff_s_bias.csv", search_path=out_dir,
                               csv_headers=
                               ["state i bits", "state j bits", "s bias mean", "std error", "ci low", "ci high"])

    # integrate the mfpt data.
    manager.integrate_csv_data(csv_name="mfpt.csv", search_path=out_dir)

    manager.close()


def main(job_args):
    run_generation(job_args.gen, job_args.submit, job_args.create_job_script, job_args.integrate_output)


# Example usage
if __name__ == "__main__":
    import argparse

    # Create the parser
    parser = argparse.ArgumentParser(

        description=

        "Script to generate sample input structures for HybridMC and submitting the job to slurm, and obtaining the "
        "output data into a database."

    )

    # Add a boolean flag (default=False). If the flag is used, the value is set to True.
    parser.add_argument('-g', '--gen', action='store_true', help="Generate structures or no")
    parser.add_argument('-c', '--create_job_script', action='store_true', help="create job script or no")
    parser.add_argument('-s', '--submit', action='store_true', help="Submit job or no")
    parser.add_argument('-i', '--integrate_output', action='store_true',
                        help="Integrate the HybridMc output data or no")

    # Parse the command-line arguments
    args = parser.parse_args()

    main(args)
