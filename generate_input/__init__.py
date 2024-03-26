import configparser
import json
import random
import os
import warnings
from functools import cached_property, partial
from typing import Any, Dict, List
import sqlite3
import subprocess
import tempfile
import csv
from string import ascii_uppercase
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from pathlib import Path


class ConfigGenerator:

    def __init__(self, base_config: Dict[str, Any] = None, num_files: int = 5, start_id=None,
                 target_directory: str = "Generated_configs", database_file_name: str = "configs.db"):

        self.base_config = base_config
        self.num_files = num_files
        self._configs = []  # To store generated configurations for uniqueness check
        self.param_settings = {}

        # set target directory
        self.target_directory = target_directory

        # Set the database name and find the id counter
        self.database_name = database_file_name
        self.id_counter = start_id

    @property
    def base_config(self) -> Dict[str, Any]:
        return self._base_config

    @base_config.setter
    def base_config(self, value: Dict[str, Any]) -> None:

        if type(value) == str:

            with open(value, 'r') as file:
                base_config = json.load(file)
            self._base_config = base_config

        elif type(value) == dict:
            self._base_config = value

        elif value is None:
            self._base_config = {}

        else:
            raise ValueError("Need a Dict or json string. Defaults to empty dictionary")

    @property
    def target_directory(self) -> str:
        return self._target_directory

    @target_directory.setter
    def target_directory(self, value: str) -> None:
        self._target_directory = value
        os.makedirs(value, exist_ok=True)

    @property
    def id_counter(self) -> int:
        return self._id_counter

    @cached_property
    def database_path(self):
        return os.path.join(self.target_directory, self.database_name)

    @id_counter.setter
    def id_counter(self, value):
        """Initialize id_counter with the last ID in the database plus one or given manual value."""
        if value:
            self._id_counter = value
        elif not os.path.exists(self.database_path):
            self._id_counter = 1
        else:
            db_manager = DatabaseManager(self.database_path)
            last_id = db_manager.get_max_id()
            self._id_counter = last_id + 1
            db_manager.close()

    def _update_database(self, rows: List[List[Any]]) -> None:
        """Update the CSV file with new configurations."""
        db_manager = DatabaseManager(self.database_path)

        # field_names =
        # Update the database with the new configurations
        db_manager.update_configurations(
            rows=rows,
            field_names=[el if el != "rc" else "nonlocal_bonds" for el in self.param_settings.keys()] + ["Filename"],
            default_values=self.base_config
        )

        # Close the database connection when done
        db_manager.close()

    def generate_configs(self, N_tries_for_unique=1000):
        return self._generate_configs(N_tries_for_unique)

    def _generate_configs(self, N_tries_for_unique) -> None:
        data = []

        # Generate new files num files number of times
        for _ in range(self.num_files):
            counter = 0
            # limit of searching 1000 times for a new configuration
            while counter < N_tries_for_unique:
                row, new_config = self.get_new_config()

                # stop looking for more configurations if the new config was found to be unique
                if new_config not in self._configs:
                    break

                else:
                    print("Trying again to find a new configuration")
                    counter += 1

            if counter > 1000:
                warnings.warn("Too many attempts to generate a new configuration. Stopping now", RuntimeWarning)
                break

            # Store new configuration in the overall configs list, so it is not duplicated
            self._configs.append(new_config)

            # add the filename to the row
            filename = f"config_{self.id_counter}.json"
            row.append(filename)

            # add the new row to the overall dataset
            data.append(row)

            #  write the new configuration to a json file for running simulations
            with open(os.path.join(self.target_directory, filename), 'w') as json_file:
                json.dump(new_config, json_file, indent=4)

            self.id_counter += 1

        # Update the database to have the new data.
        self._update_database(data)

    def get_new_config(self):
        new_config = self.base_config.copy()
        output_row = []

        # Directly update new_config and build output_row in one go
        for param, settings in self.param_settings.items():
            value = self.randomize_param(new_config, param, settings)
            output_row.append(value)

        return output_row, new_config

    def randomize_param(self, config, param, settings):
        if param == "rc":
            randomized_values = self._randomize_rcs(config["nonlocal_bonds"], settings)
            config["nonlocal_bonds"] = randomized_values
            return str(randomized_values)  # Kept as string since it is needed for output_row format

        value = None
        if 'range' in settings:
            value = random.uniform(*settings['range'])
        elif 'values' in settings:
            value = random.choice(settings['values'])

        # Update config directly here for non-"rc" params
        if value is not None:
            config[param] = value
            return value

        raise ValueError(f"No valid randomization rule found for parameter: {param}")

    @staticmethod
    def _randomize_rcs(nonlocal_bonds, settings) -> list[list[Any]]:
        if 'range' in settings:
            # pick values from a uniform distribution within the range given for each bond in nonlocal bonds

            return [[bond[0], bond[1], round(random.uniform(*settings['range']), 2)] for bond in nonlocal_bonds]

        elif 'values' in settings:
            # pick values from the set of possible choices randomly
            return [[bond[0], bond[1], random.choice(settings['values'])] for bond in nonlocal_bonds]

        else:
            raise ValueError('settings does not contain correct rc information')


class ConfigGeneratorDriver(ConfigGenerator):
    def __init__(self, settings_config_file: str = "settings.cfg"):

        config = configparser.ConfigParser()
        config.read(settings_config_file)

        # Load file master settings
        if 'master_settings' in config:
            # initialize super class with the master settings
            super().__init__(
                base_config=config.get('master_settings', 'template_structure', fallback="template.json"),
                target_directory=config.get('master_settings', 'target_directory', fallback="generated_configs"),
                database_file_name=config.get('master_settings', 'database_name', fallback="configurations.db"),
                num_files=config.getint('master_settings', 'num_files', fallback=5)
            )

            self.N_tries_for_unique = config.getint('master_settings', 'N_tries_for_unique', fallback=2000)

        # if not given then just use defaults
        else:
            super().__init__()

        # read in the param settings
        if 'param_settings' in config:
            for param, settings_str in config['param_settings'].items():
                settings_type, *values = settings_str.split(', ')
                if settings_type == 'range':
                    self.param_settings[param] = {'range': (float(values[0]), float(values[1]))}
                elif settings_type == 'values':
                    values = [json.loads(value) for value in values]
                    self.param_settings[param] = {'values': values}

    def generate_configs(self, N_tries_for_unique=None):

        # in case N_tries is given manually, use the manual value else use settings value
        if N_tries_for_unique is None:
            N_tries_for_unique = self.N_tries_for_unique

        return self._generate_configs(N_tries_for_unique)


class DatabaseManager:
    def __init__(self, db_file_name: str):
        self.db_file_name = db_file_name
        self.connection = sqlite3.connect(self.db_file_name)
        self.cursor = self.connection.cursor()

    def _ensure_table(self, field_names: List[str]):
        """Ensure the table exists and has the required columns."""
        # The ID column is defined as INTEGER PRIMARY KEY, which auto-increments.
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS configurations (ID INTEGER PRIMARY KEY)''')
        for field in field_names:
            try:
                self.cursor.execute(f'''ALTER TABLE configurations ADD COLUMN {field} TEXT''')
            except sqlite3.OperationalError:
                pass  # Column already exists

    def update_configurations(self, rows: List[List[Any]], field_names: List[str], default_values: Dict[str, Any]):
        """Update the database with new configurations, applying default values as necessary."""
        self._ensure_table(field_names)

        for row in rows:
            # Apply default values for any missing fields
            data = {field: default_values.get(field) for field in field_names}
            # Update data with actual row values
            data.update(dict(zip(field_names, row)))

            # Exclude 'ID' from the insert operation
            if 'ID' in data:
                del data['ID']

            # Prepare and execute the insert query
            columns = ', '.join(data.keys())
            placeholders = ', '.join(['?'] * len(data))
            values = tuple(data.values())
            query = f"INSERT INTO configurations ({columns}) VALUES ({placeholders})"
            self.cursor.execute(query, values)

        self.connection.commit()

    def get_max_id(self) -> int:
        """Fetch the maximum ID from the configurations table."""
        self.cursor.execute('SELECT MAX(ID) FROM configurations')
        max_id = self.cursor.fetchone()[0]
        return max_id if max_id is not None else 0

    def close(self):
        """Close the database connection."""
        self.connection.close()


class JobSubmitter:
    def __init__(self, account='def-jmschofi', job_name='get_training', cpus_per_task=4,
                 mem_per_cpu=500, time='0-01:00:00', Nconfigs="1-5", json_dir='simulation_configs',
                 out_dir="train_configs"
                 ):

        self.temp_script_path = None
        self.account = account
        self.job_name = job_name
        self.cpus_per_task = cpus_per_task
        self.mem_per_cpu = mem_per_cpu
        self.time = time
        self.Nconfigs = Nconfigs
        self.json_dir = json_dir
        self.out_dir = out_dir

    @property
    def json_dir(self) -> str:
        return self._json_dir

    @json_dir.setter
    def json_dir(self, value: str) -> None:
        self._json_dir = value
        os.makedirs(self._json_dir, exist_ok=True)

    @property
    def out_dir(self) -> str:
        return self._out_dir

    @out_dir.setter
    def out_dir(self, value: str) -> None:
        self._out_dir = value
        os.makedirs(self._out_dir, exist_ok=True)
        os.chdir(self._out_dir)
        self.json_dir = f"../{self.json_dir}"

    def create_job_script(self):
        # Create the SLURM script content
        slurm_script_content = f"""#!/bin/bash
#SBATCH --account={self.account}
#SBATCH --job-name={self.job_name}
#SBATCH --cpus-per-task={self.cpus_per_task}
#SBATCH --mem-per-cpu={self.mem_per_cpu}
#SBATCH --time={self.time}
#SBATCH --output=slurm_out/config_%A_%a.out
#SBATCH --error=slurm_err/config_%A_%a.err
#SBATCH --array={self.Nconfigs}
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=vignesh.rajesh@mail.utoronto.ca

# intitialize shell
source /home/vignesh9/.bashrc
module --force purge

# Capture start time
start_time=$(date +%s)

micromamba activate HMC

# python execute with time tracking
time hmc_run.py --json {os.path.join(self.json_dir, "config")}_"$SLURM_ARRAY_TASK_ID".json

# Capture end time and calculate duration
end_time=$(date +%s)

duration=$((end_time - start_time))
echo "Job Duration: $duration seconds"

# Additional SLURM job information
echo "Detailed job information:"
scontrol show job $SLURM_JOB_ID --details

echo "Accounting information for the job:"
sacct -j $SLURM_JOB_ID

"""

        # Write the SLURM script to a temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_script:
            temp_script.write(slurm_script_content)
            self.temp_script_path = temp_script.name

    def submit_job(self):

        # Check if job script was made, if not create one
        if self.temp_script_path is None:
            self.create_job_script()
        # Submit the SLURM job using the script
        try:
            subprocess.run(['sbatch', self.temp_script_path], check=True)
        finally:
            # Ensure the temporary file is removed after submission
            os.remove(self.temp_script_path)


class SubmitConfigJob(JobSubmitter):
    def __init__(self, settings_config_file="settings.cfg"):
        job_config = configparser.ConfigParser()
        job_config.read(settings_config_file)

        super().__init__(
            json_dir=job_config.get('master_settings', 'target_directory', fallback="generated_configs"),
            out_dir=job_config.get('slurm_settings', 'out_dir', fallback="config_run"),
            Nconfigs=job_config.get('slurm_settings', 'job_arrays', fallback="1-10"),
            cpus_per_task=job_config.getint('slurm_settings', 'n_cpus', fallback=1),
            mem_per_cpu=job_config.get('slurm_settings', 'mem_in_M', fallback=500),
            time=job_config.get('slurm_settings', 'sim_time', fallback='0-1:31:00'),
        )


class MLDatabaseManager(DatabaseManager):
    def create_ml_table(self, csv_headers: List[str]):
        csv_columns = ", ".join([f"{header} TEXT" for header in csv_headers])
        self.cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS {self.table_name} (
                "ConfigID" INTEGER,
                "nonlocal_bonds" TEXT,
                "nbeads" INTEGER,
                "Filename" TEXT,
                {csv_columns},
                FOREIGN KEY("ConfigID") REFERENCES configurations(ID)
            )
        ''')
        self.connection.commit()

    def integrate_csv_data(self, csv_name: str, search_path: str, csv_headers: List[str] = None):

        self.table_name = csv_name.replace(".csv", "")
        if csv_headers is None:
            # Defaults to the first 10 letters of the alphabet if not provided
            csv_headers = list(ascii_uppercase)[:10]

        self.cursor.execute('SELECT ID, nonlocal_bonds, nbeads, Filename FROM configurations')
        configurations = self.cursor.fetchall()

        for config in configurations:
            self.process_configuration(config, csv_headers, csv_name, search_path)

        self.connection.commit()

    def process_configuration(self, config, csv_headers, csv_name, search_path):

        config_id, nonlocal_bonds, nbeads, filename = config

        # ensure validity of filename, if not use the config_id
        if 'config' not in filename:
            filename = f"config_{config_id}.csv"
        directory = os.path.join(search_path, filename.rstrip('.json'))
        csv_file_path = os.path.join(directory, csv_name)
        if os.path.isfile(csv_file_path):
            with open(csv_file_path, 'r') as file:
                first_line = file.readline().strip().replace(" ", "")
                file.seek(0)  # Reset file pointer to the beginning

                # Check if the first line matches expected headers or is alphabetical
                is_header = all(item.isalpha() for item in first_line.split(','))

                # If detected header, use it; otherwise, use provided/default headers
                if is_header:
                    csv_reader = csv.DictReader(file)
                    dynamic_headers = csv_reader.fieldnames
                else:
                    file.seek(0)  # Reset again for csv.reader since no header row
                    csv_reader = csv.reader(file)
                    dynamic_headers = csv_headers

                # format the headers before creating the table and integrating the output rows
                dynamic_headers = [el.strip().replace(" ", "_") for el in dynamic_headers]

                self.create_ml_table(dynamic_headers)  # Ensure table matches dynamic headers

                for row in csv_reader:
                    if is_header:
                        row_values = list(row.values())
                    else:
                        row_values = row
                    self.insert_into_ml_data(config_id, nonlocal_bonds, nbeads, filename, row_values,
                                             dynamic_headers)

            print(f'File {csv_file_path} rows found. Processed.')

        else:
            print(f'File {csv_file_path} not found. Skipping.')

    def insert_into_ml_data(self, config_id, nonlocal_bonds, nbeads, filename, csv_row: List[str],
                            csv_headers: List[str]):
        """
        Function to insert into an sql lite database the simulation result values from each csv file.
        :param config_id:
        :param nonlocal_bonds:
        :param nbeads:
        :param filename:
        :param csv_row:
        :param csv_headers:
        :return:
        """

        columns = ", ".join(['ConfigID', 'nonlocal_bonds', 'nbeads', 'Filename', *csv_headers])
        placeholders = ", ".join(['?'] * (len(csv_headers) + 4))
        query = f"INSERT INTO {self.table_name} ({columns}) VALUES ({placeholders})"
        values = [config_id, nonlocal_bonds, nbeads, filename, *csv_row]
        self.cursor.execute(query, values)


class MLDatabaseManagerParallel:
    def __init__(self, output_path):
        self.output_path = Path(output_path)

    @staticmethod
    def extract_info_from_json(json_path):
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            return data.get('nonlocal_bonds'), data.get('nbeads')
        except FileNotFoundError:
            print(f"JSON file {json_path} not found!")
            return None, None

    def process_csv_and_write(self, target_directory, csv_name, json_directory, csv_headers_in_file):
        json_path = Path(json_directory) / f"{target_directory.name}.json"
        nonlocal_bonds, nbeads = self.extract_info_from_json(json_path)

        csv_file_path = target_directory / csv_name
        header = 0 if csv_headers_in_file else None

        if csv_file_path.is_file():
            for chunk in pd.read_csv(csv_file_path, chunksize=100, header=header, dtype=str):
                chunk['nonlocal_bonds'] = [nonlocal_bonds] * len(chunk)
                chunk['nbeads'] = [nbeads] * len(chunk)
                chunk['Config_Id'] = [target_directory.name.split('_')[-1]] * len(chunk)
                chunk.to_csv(self.output_path / f"all_{csv_name}", mode='a', index=False,
                             header=not csv_file_path.exists())
            print(f'Processed and wrote data from {csv_file_path}')
        else:
            print(f'CSV file not found: {csv_file_path}')

    @staticmethod
    def create_blank_csv_with_header(csv_file_path, headers):
        with open(csv_file_path, 'w', newline='') as csvfile:
            csvfile.write(','.join(headers + ["nonlocal_bonds"] + ["nbeads"] + ["Config_Id"]) + '\n')

    def integrate_csv_data(self, search_path, csv_name, csv_headers, csv_header_in_file, db_name):

        output_csv = self.output_path / f"all_{csv_name}"

        self.create_blank_csv_with_header(output_csv, csv_headers)

        directories = [d for d in Path(search_path).iterdir() if d.is_dir()]

        process_func = partial(self.process_csv_and_write,
                               csv_name=csv_name,
                               json_directory=self.output_path,
                               csv_headers_in_file=csv_header_in_file)

        with ThreadPoolExecutor() as executor:
            list(executor.map(process_func, directories))

        self.insert_csv_into_sqlite(db_path=self.output_path / db_name,
                                    csv_path=output_csv,
                                    table_name=csv_name.replace('.csv', ''),
                                    if_exists='replace')

    @staticmethod
    def insert_csv_into_sqlite(db_path, csv_path, table_name, if_exists='replace'):
        """
        Inserts data from a CSV file into a SQLite database table.

        :param db_path: Path to the SQLite database file.
        :param csv_path: Path to the CSV file to be inserted.
        :param table_name: Name of the table where the CSV data will be inserted.
        :param if_exists: Action to take if the table already exists. Options are 'fail', 'replace', or 'append'.
        """
        # Establish a connection to the SQLite database
        with sqlite3.connect(db_path) as conn:
            # Load the CSV file into a pandas DataFrame
            df = pd.read_csv(csv_path)

            # Replace spaces with underscores in DataFrame column names
            df.columns = [col.replace(' ', '_') for col in df.columns]

            # Insert the DataFrame into the SQLite database
            df.to_sql(name=table_name, con=conn, if_exists=if_exists, index=False)


class EnhancedConfigGenerator(ConfigGenerator):
    def randomize_param(self, config, param, settings):
        # Special handling for 'nonlocal_bonds' parameter
        if param == "nonlocal_bonds":
            if 'nbeads' in config:
                nbeads = config['nbeads']
                # Generate a random index1 between 0 and nbeads-1
                index1 = random.randint(0, nbeads - 1)

                # Calculate allowed values for index2, excluding nearest and next-nearest neighbors
                excluded_indices = {(index1 + i) % nbeads for i in (-2, -1, 0, 1, 2)}
                allowed_indices = set(range(nbeads)) - excluded_indices

                # If no allowed indices are available (which may happen in small systems), raise an error
                if not allowed_indices:
                    raise ValueError(
                        "Cannot find a suitable index2 that is not a nearest or next-nearest neighbor of index1 due to small nbeads.")

                # Choose index2 from the allowed values
                index2 = random.choice(list(allowed_indices))

                # Assuming nonlocal_bonds expects a list of tuples or lists with indices
                config[param] = [[index1, index2]]
                return str([[index1, index2]])  # Convert to string if needed for output_row format
            else:
                raise ValueError(
                    "nbeads value is required for randomizing nonlocal_bonds indices but not found in config")

        # For all other parameters, use the base class method
        return super().randomize_param(config, param, settings)