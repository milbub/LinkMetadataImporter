from mariadb import Cursor
from datetime import datetime
from pathlib import Path
import sys
import pandas as pd
import mariadb
import configparser
import logging
import re


"""
This script is used for import of CML metadata of specific CML ISP into the Telcorain's SQL database structure.
This script IS NOT UNIVERSAL in any way nor is designed like that.
"""


'''
Start logging setup
'''

# create a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# create a file handler and set its level
t = datetime.now()
Path("./logs").mkdir(parents=True, exist_ok=True)
file_handler = logging.FileHandler(f'./logs/IMPORT_{t.year}-{t.month:02d}-{t.day:02d}_'
                                   f'{t.hour:02d}-{t.minute:02d}-{t.second:02d}.log')
file_handler.setLevel(logging.DEBUG)

# create a console handler and set its level
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# create a formatter
formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')

# set formatter to handlers
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# add the handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

'''
End logging setup
'''


def check_path(path):
    if path == "":
        error = f"ERROR: Cannot start! Path is empty!"
        logger.error(error)
        raise FileNotFoundError(error)

    p = Path(path)
    if p.exists():
        logger.info(f"Using file: {p.absolute()}")
    else:
        error = f"ERROR: Cannot start! File: {path} is missing!"
        logger.error(error)
        raise FileNotFoundError(error)


def load_db_settings(path) -> {}:
    def read_option(parser, option) -> str:
        if not parser.has_option('mariadb', option):
            error = "ERROR: Missing option in DB configuration file. Check the config!"
            logger.error(error)
            raise ModuleNotFoundError(error)

        return parser['mariadb'][option]

    p = configparser.ConfigParser()
    p.read(path, encoding='utf-8')

    settings = {
        'address': read_option(p, 'address'),
        'port': read_option(p, 'port'),
        'user': read_option(p, 'user'),
        'pass': read_option(p, 'pass'),
        'timeout': read_option(p, 'timeout'),
        'db_metadata': read_option(p, 'db_metadata'),
    }

    return settings


def check_db_connection(db_connection):
    try:
        db_connection.ping()
    except mariadb.InterfaceError:
        error = "ERROR: Cannot connect to the MariaDB."
        logger.error(error)
        raise ConnectionError()


def filter_data(df: pd.DataFrame) -> pd.DataFrame:
    # rule 1: filter 'status' column
    df = df[df['status'].isin(['V provozu', 'Upgrade proveden', 'Provést upgrade'])]

    # rule 2: filter 'technologie_Nis' column to keep rows with valid integer numbers
    df = df[pd.to_numeric(df['technologie_Nis'], errors='coerce').notnull()]

    # rule 3: filter rows with valid IPv4 addresses in 'A IP adresa' and 'B IP adresa'
    ip_pattern = re.compile(r'^(?:[0-9]{1,3}\.){3}[0-9]{1,3}(?::[0-9]{1,5})?$')
    df = df[df['A IP adresa'].str.match(ip_pattern) & df['B IP adresa'].str.match(ip_pattern)]

    # helper function to process frequency columns
    def process_frequency_columns(row, src_col, dest_col):
        value = row[src_col]
        if pd.isnull(value):
            value = row[dest_col]
        if pd.isnull(value) and row['pasmo'] == '10,5':
            return 10500.0
        return value

    # rule 4: filter frequencies of A units
    df['NAST_A_frq'] = df.apply(process_frequency_columns, src_col='Aktuální upgrade::A_rf_frekvence',
                                dest_col='NAST_A_frq', axis=1)

    # rule 5: filter frequencies of B units
    df['NAST_B_frq'] = df.apply(process_frequency_columns, src_col='Aktuální upgrade::B_rf_frekvence',
                                dest_col='NAST_B_frq', axis=1)

    # helper function to process polarization columns
    def process_polarization(row, src_col, dest_col):
        mapping = {
            'vertikální': 'V', 'Vertikální': 'V', 'vertikalni': 'V', 'vertical': 'V',
            'horizontální': 'H', 'Horizontální': 'H', 'horizontalni': 'H', 'horizontal': 'H',
            'XPIC': 'X', 'V + H': 'X', 'V+H': 'X'
        }
        value = row[src_col]
        if pd.isnull(value):
            value = row[dest_col]
        return mapping.get(value, value)

    # rule 6: filter polarizations
    df['NAST_A_polarizace'] = df.apply(process_polarization, src_col='Aktuální upgrade::A_rf_polarizace',
                                       dest_col='NAST_A_polarizace', axis=1)

    # determine which ID column to use
    id_column = '__pk_ID' if '__pk_ID' in df.columns else '__pk_ID_new_calc'

    # before filling NaN, get the IDs of the rows where 'NAST_A_polarizace' is NaN for logging
    rows_to_fill = df[df['NAST_A_polarizace'].isna()][id_column]
    for id_value in rows_to_fill:
        logger.debug(f"The CML {id_value} does not have filled polarization type. It has been filled with default "
                     f"vertical polarization (V).")

    # default polarization is vertical
    df['NAST_A_polarizace'] = df['NAST_A_polarizace'].fillna('V')

    # drop rows where frequency columns are null
    df = df.dropna(subset=['NAST_A_frq', 'NAST_B_frq'])

    return df


if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) >= 3:
        cbl_xlsx_path = args[0]
        mcl_xlsx_path = args[1]
        corrs_xlsx_path = args[2]

        if len(args) >= 4:
            db_config_path = args[3]
        else:
            db_config_path = './database.ini'
    else:
        cbl_xlsx_path = input("Please enter a full path to the CBL's exported XLSX file: ")
        cbl_xlsx_path = cbl_xlsx_path.replace('"', '').replace("'", "")

        mcl_xlsx_path = input("Please enter a full path to the MCL's exported XLSX file: ")
        mcl_xlsx_path = mcl_xlsx_path.replace('"', '').replace("'", "")

        corrs_xlsx_path = input("Please enter a full path to a corrections XLSX file: ")
        corrs_xlsx_path = corrs_xlsx_path.replace('"', '').replace("'", "")

        db_config_path = input("Default DB configuration file's name is 'database.ini' and is located in the same "
                               "directory, as this script.\nIn case of different name or path, please enter a full path"
                               " to this file, or left it empty: ")
        if db_config_path == '':
            db_config_path = './database.ini'
        else:
            db_config_path = db_config_path.replace('"', '').replace("'", "")

    # check if files exist
    check_path(cbl_xlsx_path)
    check_path(mcl_xlsx_path)
    check_path(corrs_xlsx_path)
    check_path(db_config_path)

    # load connection settings
    db_settings = load_db_settings(db_config_path)

    # create and test DB connection
    connection = mariadb.connect(
        user=db_settings['user'],
        password=db_settings['pass'],
        host=db_settings['address'],
        port=int(db_settings['port']),
        database=db_settings['db_metadata'],
        connect_timeout=int(int(db_settings['timeout']) / 1000),
        reconnect=True
    )
    check_db_connection(connection)
    logger.info("**********************************************")
    logger.info("Connection to MariaDB: OK")
    logger.info("**********************************************")

    # load XLSXs and parse them into dataframes
    logger.info("Loading CBL XLSX...")
    cbl_xlsx = pd.ExcelFile(cbl_xlsx_path)
    cbl_df = cbl_xlsx.parse('Sheet1')
    cbl_orig_count = cbl_df.shape[0]
    logger.info(f"OK. Total of {cbl_orig_count} CBL rows loaded.")

    logger.info(f"Loading MCL XLSX...")
    mcl_xlsx = pd.ExcelFile(mcl_xlsx_path)
    mcl_df = mcl_xlsx.parse('Sheet1')
    mcl_orig_count = mcl_df.shape[0]
    logger.info(f"OK. Total of {mcl_orig_count} MCL rows loaded.")

    logger.info(f"Loading corrections XLSX...")
    corrs_xlsx = pd.ExcelFile(corrs_xlsx_path)
    corrs_cbl_df = corrs_xlsx.parse('CBL')
    corrs_mcl_df = corrs_xlsx.parse('MCL')
    logger.info(f"OK. Total of {corrs_cbl_df.shape[0]} corrected CBL "
                f"and {corrs_mcl_df.shape[0]} corrected MCL rows loaded.")

    # filter data
    logger.info("**********************************************")
    logger.info("Starting filtering CBL data...")
    cbl_df = filter_data(cbl_df)
    logger.info(f"CBL data filtering ended. {cbl_df.shape[0]} rows remained, {cbl_orig_count - cbl_df.shape[0]} rows "
                f"dropped.")

    logger.info("Starting filtering MCL data...")
    mcl_df = filter_data(mcl_df)
    logger.info(f"MCL data filtering ended. {mcl_df.shape[0]} rows remained, {mcl_orig_count - mcl_df.shape[0]} rows "
                f"dropped.")
    logger.info("**********************************************")







