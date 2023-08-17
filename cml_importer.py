from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import sys
import requests
import mariadb
import configparser
import logging
import re
import random
import math


"""
DISCLAIMER:
This script is used for import of CML metadata of specific CML ISP into the Telcorain's SQL database structure.
This script IS NOT UNIVERSAL in any way nor is designed like that.
"""

"""
START CONFIGURATION OPTIONS
"""

FETCH_ALT_FROM_GMAPS = True

AREA_BORDER_X_MIN = 11  # MIN X coordinate of processed area in degrees (now Czechia with surrounding border regions)
AREA_BORDER_X_MAX = 20  # MAX X coordinate of processed area in degrees (now Czechia with surrounding border regions)
AREA_BORDER_Y_MIN = 48  # MIN Y coordinate of processed area in degrees (now Czechia with surrounding border regions)
AREA_BORDER_Y_MAX = 52  # MAX Y coordinate of processed area in degrees (now Czechia with surrounding border regions)

COORD_CHECK_TOLERANCE = 0.001  # tolerance for duplicity coordinates check in degrees
HEIGHT_CHECK_TOLERANCE = 5  # tolerance for duplicity height check in meters

RANDOM_SHIFT_MIN = 500  # minimum radius of dummy coordinate shift in meters
RANDOM_SHIFT_MAX = 1500  # maximum radius of dummy coordinate shift in meters

"""
END CONFIGURATION OPTIONS
"""

'''
START LOGGING SETUP
'''

# create a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# create a file handler and set its level
t = datetime.now()
Path("./logs").mkdir(parents=True, exist_ok=True)
file_handler = logging.FileHandler(f'./logs/IMPORT_{t.year}-{t.month:02d}-{t.day:02d}_'
                                   f'{t.hour:02d}-{t.minute:02d}-{t.second:02d}.log',
                                   'w', encoding='utf-8')
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
END LOGGING SETUP
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


def load_settings(path) -> {}:
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
        'gmaps_api_key': read_option(p, 'gmaps_api_key')
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


def apply_corrections(main_df: pd.DataFrame, corrections_df: pd.DataFrame) -> pd.DataFrame:
    # determine which ID column to use
    id_column = '__pk_ID' if '__pk_ID' in main_df.columns else '__pk_ID_new_calc'

    # for each row in corrections_df
    for _, row in corrections_df.iterrows():
        # get the unique ID from the row
        pk_id = row[id_column]

        # for each column in corrections_df, excluding 'KOMENTÁŘ'
        for col in corrections_df.columns:
            if col != 'KOMENTÁŘ' and pd.notna(row[col]):
                # update the main_df with the value from corrections_df
                main_df.loc[main_df[id_column] == pk_id, col] = row[col]

                logger.debug(f"Applied correction of '{col}' with '{row[col]}' for CML {pk_id}.")
    return main_df


def drop_duplicate_ips(df: pd.DataFrame) -> pd.DataFrame:
    # determine which ID column to use
    id_column = '__pk_ID' if '__pk_ID' in df.columns else '__pk_ID_new_calc'

    # combine the IPs from 'A IP adresa' and 'B IP adresa'
    combined_ips = df['A IP adresa'].tolist() + df['B IP adresa'].tolist()

    # convert combined list into a set to get unique IPs
    unique_ips = set(combined_ips)

    # counter for dropped rows
    dropped_count = 0

    # for each unique IP, identify rows where it occurs in 'A IP adresa' or 'B IP adresa'
    for ip in unique_ips:
        # get all rows where IP is in 'A IP adresa' or 'B IP adresa'
        duplicate_rows = df[(df['A IP adresa'] == ip) | (df['B IP adresa'] == ip)]

        # if there's more than one row with this IP
        if len(duplicate_rows) > 1:
            # sort the rows by '__pk_ID' in descending order and drop all but the first row
            duplicate_rows = duplicate_rows.sort_values(by=id_column, ascending=False)

            # print the __pk_IDs of the rows being dropped
            for dropped_id in duplicate_rows.iloc[1:][id_column]:
                logger.debug(f"Dropped CML {dropped_id} doe to its duplicate IP address {ip}.")
                dropped_count += 1

            # drop the rows from the main dataframe
            df = df.drop(duplicate_rows.iloc[1:].index)

    logger.info(f"Total of {dropped_count} rows were dropped due to duplicate IP.")

    return df


def merge_datasets(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    # list of columns to keep
    columns_to_keep = [
        'název', 'status', 'souradnice_vzd', 'souradnice_B1', 'souradnice_B2',
        'souradnice_B3', 'souradnice_A1', 'souradnice_A2', 'souradnice_A3',
        'souradnice_B4', 'souradnice_B5', 'souradnice_B6', 'souradnice_A4',
        'souradnice_A5', 'souradnice_A6', 'souradnice_A dec1', 'souradnice_A dec2',
        'souradnice_B dec1', 'souradnice_B dec2', 'souradnice_A_vyska nad terenem',
        'souradnice_B_vyska nad terenem', 'souradnice_A dec_komplet', 'souradnice_B dec_komplet',
        'A IP adresa', 'B IP adresa', 'technologie_Nis', 'NAST_A_frq', 'NAST_B_frq', 'NAST_A_polarizace'
    ]

    if '__pk_ID_new_calc' in df1.columns:
        df1 = df1.rename(columns={'__pk_ID_new_calc': '__pk_ID'})

    if '__pk_ID_new_calc' in df2.columns:
        df2 = df2.rename(columns={'__pk_ID_new_calc': '__pk_ID'})

    df1 = df1[columns_to_keep + ['__pk_ID']]
    df2 = df2[columns_to_keep + ['__pk_ID']]

    mg_df = pd.concat([df1, df2], ignore_index=True)

    return mg_df


def process_coordinates(input_df: pd.DataFrame) -> pd.DataFrame:
    # helper function to convert DMS to decimal
    def dms_to_decimal(degrees, minutes, seconds):
        return degrees + minutes / 60.0 + seconds / 3600.0

    msgs_no_data = []
    msgs_broken_coord = []
    msgs_swapped_coord = []
    msgs_invalid_coord = []
    msgs_conflict_coord = []
    msgs_conflict_height = []

    sites_list = []

    for _, row in input_df.iterrows():
        for site_type in ['A', 'B']:

            if site_type == 'A':
                address = row['název'].split(' - ')[0].strip()
                pk_id = row['__pk_ID']
                x_dec = row['souradnice_A dec2']
                y_dec = row['souradnice_A dec1']
                x_deg = row['souradnice_A4']
                x_min = row['souradnice_A5']
                x_sec = row['souradnice_A6']
                y_deg = row['souradnice_A1']
                y_min = row['souradnice_A2']
                y_sec = row['souradnice_A3']
                height = row['souradnice_A_vyska nad terenem']

            else:
                address = row['název'].split(' - ')[1].strip()
                pk_id = row['__pk_ID']
                x_dec = row['souradnice_B dec2']
                y_dec = row['souradnice_B dec1']
                x_deg = row['souradnice_B4']
                x_min = row['souradnice_B5']
                x_sec = row['souradnice_B6']
                y_deg = row['souradnice_B1']
                y_min = row['souradnice_B2']
                y_sec = row['souradnice_B3']
                height = row['souradnice_B_vyska nad terenem']

            # extract city from address
            city = address.split('_')[0] if '_' in address else np.nan

            # compute X and Y coordinates
            if pd.isnull(x_dec):
                if all(pd.notnull([x_deg, x_min, x_sec])):
                    x_dec = dms_to_decimal(x_deg, x_min, x_sec)
                else:
                    msgs_no_data.append(f"NO DATA for address: {address} on CML: {pk_id}")
                    continue

            if pd.isnull(y_dec):
                if all(pd.notnull([y_deg, y_min, y_sec])):
                    y_dec = dms_to_decimal(y_deg, y_min, y_sec)
                else:
                    msgs_no_data.append(f"NO DATA for address: {address} on CML: {pk_id}")
                    continue

            # helper function for correcting the decimal point in some weirdly formatted source coordinates
            def fix_decimal_after_two_digits(num):
                s = str(int(num))  # Convert to string, removing any decimal if present
                return float(s[:2] + '.' + s[2:])

            if x_dec > 180:
                msgs_broken_coord.append(f"X coordinate of CML: {pk_id} is broken (X: {x_dec}, fixing decimal point.)")
                x_dec = fix_decimal_after_two_digits(x_dec)
            if y_dec > 90:
                msgs_broken_coord.append(f"Y coordinate of CML: {pk_id} is broken (Y: {y_dec}, fixing decimal point.)")
                y_dec = fix_decimal_after_two_digits(y_dec)

            # check for swapped coordinates
            if (x_dec < AREA_BORDER_X_MIN) or (x_dec > AREA_BORDER_X_MAX):
                if y_dec < AREA_BORDER_Y_MIN:
                    x_dec, y_dec = y_dec, x_dec
                    msgs_swapped_coord.append(f"Swapped coordinates on CML: {pk_id}! Corrected.")
                else:
                    msgs_invalid_coord.append(f"Invalid X coordinate (X: {x_dec}, Y: {y_dec}) on CML: {pk_id}. "
                                              f"Skipping.")
                    continue

            # check for invalid coordinate
            if (y_dec < AREA_BORDER_Y_MIN) or (y_dec > AREA_BORDER_Y_MAX):
                msgs_invalid_coord.append(f"Invalid Y coordinate (X: {x_dec}, Y: {y_dec}) on CML: {pk_id}. Skipping.")
                continue

            existing_site = next((site for site in sites_list
                                  if site['address'].lower().strip() == address.lower().strip()), None)

            if existing_site:
                # check for data conflict
                diff_x = round(abs(existing_site['X_coordinate'] - x_dec), 3)
                diff_y = round(abs(existing_site['Y_coordinate'] - y_dec), 3)
                if (diff_x > COORD_CHECK_TOLERANCE) or (diff_y > COORD_CHECK_TOLERANCE):
                    msgs_conflict_coord.append(f"COORDS CONFLICT (diff: {diff_x}/{diff_y}) at address: {address}. "
                                               f"Original (X: {existing_site['X_coordinate']}, "
                                               f"Y: {existing_site['Y_coordinate']}). "
                                               f"New (X: {x_dec}, Y: {y_dec}) on CML: {pk_id}")

                # check and update height_above_terrain
                if pd.isnull(existing_site['height_above_terrain']) and pd.notnull(height):
                    existing_site['height_above_terrain'] = height
                    # logger.debug(f"Updated 'height_above_terrain' for address: {address} to value: "
                    #              f"{height} m from CML: {pk_id}.")
                elif pd.notnull(existing_site['height_above_terrain']) and pd.notnull(height) and \
                        (abs(existing_site['height_above_terrain'] - height) > HEIGHT_CHECK_TOLERANCE):
                    msgs_conflict_height.append(f"HEIGHT CONFLICT at address: {address}. "
                                                f"Original: {existing_site['height_above_terrain']} m. "
                                                f"New: {height} m on CML: {pk_id}.")
            else:
                sites_list.append({
                    'address': address,
                    'city': city,
                    'X_coordinate': x_dec,
                    'Y_coordinate': y_dec,
                    'X_dummy_coordinate': np.nan,
                    'Y_dummy_coordinate': np.nan,
                    'height_above_terrain': height
                })

    st_df = pd.DataFrame(sites_list)
    st_df.set_index('address', inplace=True)

    for msgs in [msgs_no_data, msgs_broken_coord, msgs_invalid_coord,
                 msgs_swapped_coord, msgs_conflict_coord, msgs_conflict_height]:
        for msg in msgs:
            logger.debug(msg)

    return st_df


def fetch_altitude(lat, lon, api_key) -> float | None:
    if FETCH_ALT_FROM_GMAPS:
        try:
            url = f"https://maps.googleapis.com/maps/api/elevation/json?locations={lat},{lon}&key={api_key}"
            response = requests.get(url)
            result = response.json()

            if result["status"] == "OK":
                return round(result["results"][0]["elevation"], 2)
            else:
                raise ConnectionError(f"Error fetching elevation for {lat}, {lon}. Error: {result['status']}")
        except BaseException as e:
            logger.error(e)
            return None
    else:
        return None


def calc_dummy_coordinates(x, y) -> (float, float):
    radius = random.uniform(RANDOM_SHIFT_MIN, RANDOM_SHIFT_MAX) / 1000  # convert to km for calculation
    angle = random.uniform(0, 2 * math.pi)
    x_dummy = x + radius * math.cos(angle) / 111.32  # rough conversion of km to degrees
    y_dummy = y + radius * math.sin(angle) / (111.32 * math.cos(x * math.pi / 180))

    return round(x_dummy, 8), round(y_dummy, 8)


def update_sites(sites_dataset: pd.DataFrame, conn, cursor, GMAPS_API_KEY) -> (int, int):
    # fetch all records from the sites table
    cursor.execute("SELECT address, X_coordinate, Y_coordinate, height_above_terrain FROM sites")
    db_sites = cursor.fetchall()

    # fetch column names from cursor description
    columns = [desc[0] for desc in cursor.description]

    # convert DB results to dictionary
    db_sites_dict = {}
    for site in db_sites:
        site_dict = dict(zip(columns, site))
        db_sites_dict[site_dict['address']] = site_dict

    update_counter, insert_counter = 0, 0

    for _, row in sites_dataset.iterrows():
        address = _
        city = row['city']
        x = round(row['X_coordinate'], 8)
        y = round(row['Y_coordinate'], 8)
        height = row['height_above_terrain']

        if np.isnan(x) or np.isnan(y):
            continue

        if pd.isna(city):
            city = None

        if np.isnan(height):
            height = None

        if address in db_sites_dict:
            site = db_sites_dict[address]

            if round(site['X_coordinate'], 8) != x or round(site['Y_coordinate'], 8) != y \
                    or site['height_above_terrain'] != height:
                # get dummy coords
                x_dummy, y_dummy = calc_dummy_coordinates(x, y)
                # get altitude
                altitude = fetch_altitude(y, x, GMAPS_API_KEY)

                # update the record
                sql = """UPDATE sites SET X_coordinate = %s, Y_coordinate = %s, height_above_terrain = %s, 
                                           X_dummy_coordinate = %s, Y_dummy_coordinate = %s, altitude = %s 
                        WHERE address = %s"""
                cursor.execute(sql, (x, y, height, x_dummy, y_dummy, altitude, address))
                logger.debug(f"Updated site: {address} in DB with new coordinates (X: {x}, Y: {y}) "
                             f"or height ({height} m).")
                update_counter += 1

        else:
            # get dummy coords
            x_dummy, y_dummy = calc_dummy_coordinates(x, y)
            # get altitude
            altitude = fetch_altitude(y, x, GMAPS_API_KEY)

            # insert the new record
            sql = """INSERT INTO sites (address, city, X_coordinate, Y_coordinate, height_above_terrain, 
                                         X_dummy_coordinate, Y_dummy_coordinate, altitude) 
                     VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"""
            cursor.execute(sql, (address, city, x, y, height, x_dummy, y_dummy, altitude))
            logger.debug(f"Inserted new site: {address} into the DB.")
            insert_counter += 1

    conn.commit()
    return update_counter, insert_counter


def update_links(input_df, conn, cursor) -> (float, float, float, float):
    # splitting 'název' column to sites A and B
    input_df['site_A'] = input_df['název'].str.split(' - ').str[0].str.strip()
    input_df['site_B'] = input_df['název'].str.split(' - ').str[1].str.strip()

    # remove 'MW' from IDs and convert them into integers
    input_df['__pk_ID'] = input_df['__pk_ID'].str[2:].astype(int)

    # remove port numbers from IP addresses
    input_df['A IP adresa'] = input_df['A IP adresa'].str.split(':').str[0]
    input_df['B IP adresa'] = input_df['B IP adresa'].str.split(':').str[0]

    # querying the 'links' table and joining with 'sites' and 'technologies'
    cursor.execute("""
        SELECT 
            links.ISP_ID, 
            links.is_active, 
            links.IP_address_A, 
            links.IP_address_B,
            links.frequency_A, 
            links.frequency_B, 
            links.polarization,
            A.address as site_A_address, 
            B.address as site_B_address,
            tech.ISP_ID as technology_ISP_ID 
        FROM links
        JOIN sites A ON links.site_A = A.ID
        JOIN sites B ON links.site_B = B.ID
        JOIN technologies tech ON links.technology = tech.ID
    """)

    db_links = cursor.fetchall()

    # fetch column names from cursor description
    columns = [desc[0] for desc in cursor.description]

    # convert DB results to dictionary
    links_dict = {}
    for link in db_links:
        link_dict = dict(zip(columns, link))
        links_dict[link_dict['ISP_ID']] = link_dict

    counter_skipped, counter_updated, counter_inserted, counter_deactivated = 0, 0, 0, 0

    for _, row in input_df.iterrows():
        current_id = row['__pk_ID']

        # retrieve IDs from 'sites' and 'technologies' tables
        cursor.execute("SELECT ID FROM sites WHERE address = %s", (row['site_A'],))
        fetched_A_id = cursor.fetchone()
        if fetched_A_id is not None:
            site_A_id = fetched_A_id[0]
        else:
            logger.debug(f"Cannot find address '{row['site_A']}' in DB for CML: {current_id}, skipping CML.")
            counter_skipped += 1
            continue

        cursor.execute("SELECT ID FROM sites WHERE address = %s", (row['site_B'],))
        fetched_B_id = cursor.fetchone()
        if fetched_B_id is not None:
            site_B_id = fetched_B_id[0]
        else:
            logger.debug(f"Cannot find address '{row['site_B']}' in DB for CML: {current_id}, skipping CML.")
            counter_skipped += 1
            continue

        cursor.execute("SELECT ID FROM technologies WHERE ISP_ID = %s", (row['technologie_Nis'],))
        fetched_tech_id = cursor.fetchone()
        if fetched_tech_id is not None:
            technology_id = fetched_tech_id[0]
        else:
            logger.debug(f"Cannot find technology {row['technologie_Nis']} in DB for CML: {current_id}, skipping CML.")
            counter_skipped += 1
            continue

        # if current ID exists in db
        if current_id in links_dict:
            link_data = links_dict[current_id]

            # check differences and update if needed
            updates = []
            if link_data['is_active'] != 1:
                updates.append("is_active = 1")
            if link_data['IP_address_A'] != row['A IP adresa']:
                updates.append("IP_address_A = '{}'".format(row['A IP adresa']))
            if link_data['IP_address_B'] != row['B IP adresa']:
                updates.append("IP_address_B = '{}'".format(row['B IP adresa']))
            if link_data['frequency_A'] != row['NAST_A_frq']:
                updates.append("frequency_A = '{}'".format(row['NAST_A_frq']))
            if link_data['frequency_B'] != row['NAST_B_frq']:
                updates.append("frequency_B = '{}'".format(row['NAST_B_frq']))
            if link_data['polarization'] != row['NAST_A_polarizace']:
                updates.append("polarization = '{}'".format(row['NAST_A_polarizace']))
            if link_data['site_A_address'] != row['site_A']:
                updates.append("site_A = {}".format(site_A_id))
            if link_data['site_B_address'] != row['site_B']:
                updates.append("site_B = {}".format(site_B_id))
            if link_data['technology_ISP_ID'] != row['technologie_Nis']:
                updates.append("technology = {}".format(technology_id))

            if updates:
                update_query = "UPDATE links SET {} WHERE ISP_ID = %s".format(", ".join(updates))
                cursor.execute(update_query, (current_id,))

                # update the modify_time
                cursor.execute("UPDATE links SET modify_time = %s WHERE ISP_ID = %s", (datetime.utcnow(), current_id))

                logger.debug(f"Updated CML: {current_id} with new data.")
                counter_updated += 1

            # remove the processed row from links_dict
            del links_dict[current_id]

        else:
            # insert a new row into links table
            cursor.execute("""
                INSERT INTO links (ISP_ID, is_active, IP_address_A, IP_address_B, frequency_A, frequency_B, 
                polarization, site_A, site_B, technology, import_time)
                VALUES (%s, 1, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (current_id, row['A IP adresa'], row['B IP adresa'], row['NAST_A_frq'], row['NAST_B_frq'],
                  row['NAST_A_polarizace'], site_A_id, site_B_id, technology_id, datetime.utcnow()))

            logger.debug(f"Inserted CML: {current_id} into the DB.")
            counter_inserted += 1

    # update remaining rows in links_dict as inactive
    for remaining_id in links_dict.keys():
        cursor.execute("""UPDATE links SET is_active = 0 WHERE ISP_ID = %s""", (remaining_id,))

        # update the modify_time
        cursor.execute("UPDATE links SET modify_time = %s WHERE ISP_ID = %s", (datetime.utcnow(), remaining_id))

        counter_deactivated += 1

    conn.commit()

    return counter_skipped, counter_updated, counter_inserted, counter_deactivated


if __name__ == '__main__':
    args = sys.argv[1:]

    # if arguments are provided, use them. If not, user will be prompted for the input.
    # arguments are in order: CBL xlsx, MCL xlsx, corrections xlsx, DB INI config (optional, default is root dir)
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

    # load settings
    settings = load_settings(db_config_path)

    # create and test DB connection
    connection = mariadb.connect(
        user=settings['user'],
        password=settings['pass'],
        host=settings['address'],
        port=int(settings['port']),
        database=settings['db_metadata'],
        connect_timeout=int(int(settings['timeout']) / 1000),
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

    # apply corrections
    logger.info("Correcting CBL data...")
    cbl_df = apply_corrections(cbl_df, corrs_cbl_df)
    logger.info("OK.")

    logger.info("Correcting MCL data...")
    mcl_df = apply_corrections(mcl_df, corrs_mcl_df)
    logger.info("OK.")

    # remove duplicate IPs
    logger.info("**********************************************")
    logger.info("Remove duplicate IPs from CBL data...")
    cbl_df = drop_duplicate_ips(cbl_df)
    logger.info("Remove duplicate IPs from MCL data...")
    mcl_df = drop_duplicate_ips(mcl_df)
    logger.info("**********************************************")

    # merge CBL and MCL datasets before processing coordinates
    logger.info("Merging CBL and MCL datasets before processing coordinates...")
    merged_df = merge_datasets(cbl_df, mcl_df)
    logger.info("OK.")

    # process coordinates
    logger.info("Processing coordinates and creating sites list...")
    sites_df = process_coordinates(merged_df)
    logger.info("OK.")

    # update sites in DB and fetch altitude if needed
    logger.info("**********************************************")
    gmaps_api_key = settings['gmaps_api_key']
    db_cursor = connection.cursor()
    logger.info("Updating sites in DB...")
    updated, inserted = update_sites(sites_df, connection, db_cursor, gmaps_api_key)
    logger.info(f"OK. {updated} existing sites has been updated, {inserted} new sites has been inserted.")
    logger.info("**********************************************")

    # update links in DB
    logger.info("Updating links in DB...")
    skipped, updated, inserted, deactivated = update_links(merged_df, connection, db_cursor)
    logger.info(f"OK. {skipped} CMLs has been skipped due invalid data. {updated} CMLs has been updated, {inserted} "
                f"new CMLs has been inserted. {deactivated} CMLs has been deactivated.")
    logger.info(f"Processing done. END.")
    connection.close()
