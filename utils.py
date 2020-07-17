import srtm, srtm.main
import gpxpy
import re
from datetime import timedelta
import os
import urllib.request
import zipfile
import numpy as np
import pandas as pd
from IPython.display import clear_output

ZIP_FILE = '32068_41607_compressed_gpx-tracks-from-hikr.org.csv.zip'
CSV_FILE = 'gpx-tracks-from-hikr.org.csv'
DATA_FILE = 'tracks.pkl'
URL = 'https://www.kaggle.com/roccoli/gpx-hike-tracks'

def drop_where(df, condition):
    '''
    Rimuove tutte le righe dal DataFrame `df` per le quali la Series `condition` è `True`.
    Gli indici devono essere compatibili.
    L operazione viene fatta "in place".
    '''
    df.drop(df[condition].index, inplace=True)

def has_all_elevations(gpx):
    ''' Controlla che il gpx abbia i dati di altitudine in tutti i punti '''
    for point in gpx.walk(only_points=True):
        if point.elevation is None:
            return False
    return True

class LocalSRTMFileHandler(srtm.main.FileHandler):
    def __init__(self, cache_dir):
        self.dir = cache_dir
        if not os.path.isdir(self.dir):
            os.mkdir(self.dir)
        
    def get_srtm_dir(self):
        return self.dir

def clean_data(data):
    # Conversione dei campi temporali
    data['moving_time'] = pd.to_timedelta(data['moving_time'], unit='s')
    time_cols = ['start_time', 'end_time']
    for col in time_cols:
        data[col] = pd.to_datetime(data[col], infer_datetime_format=True, errors='coerce')
    
    # Per i record in cui 'moving_time' è zero,
    # si cerca di calcolarlo come differenza tra 'end_time' e 'start_time'
    no_moving = data['moving_time'] == timedelta(0)
    data.loc[no_moving, 'moving_time'] = data.loc[no_moving, 'end_time'] - data.loc[no_moving, 'start_time']
    
    # Rimozione righe che non hanno un valore di 'moving_time' valido e positivo
    drop_where(data, (data['moving_time'] <= timedelta(0)) | data['moving_time'].isnull())
    
    # Rimozione righe per cui la velocità media è > 15km/h
    hours = data['moving_time'] / timedelta(hours=1)
    km = data['length_2d'] / 1000
    speed = km / hours
    drop_where(data, speed > 15)
    
    # Rimozione righe con 'length_2d' < 1000
    drop_where(data, data['length_2d'] < 1000)
    
    # Ricalcolo altitudine
    gpx_ok = pd.Series(True, index=data.index)
    elevation_data = srtm.get_data(batch_mode=True, file_handler=LocalSRTMFileHandler('srtm_cache'))
    total = len(data)
    for linei, idx in enumerate(data.index):
        print(f'Ricalcolo altitudine {linei+1}/{total}', end='\r')
        try:
            gpx = gpxpy.parse(data.loc[idx, 'gpx'])
        except:
            gpx_ok.loc[idx] = False
        else:
            gpx_ok.loc[idx] = has_all_elevations(gpx)
            retry = True
            while retry:
                retry = False
                try:
                    elevation_data.add_elevations(gpx, smooth=True)
                    gpx.add_missing_elevations()
                except ReadTimeoutError:
                    retry = True
            if has_all_elevations(gpx):
                data.loc[idx, ['min_elevation', 'max_elevation']] = tuple(gpx.get_elevation_extremes())
                data.loc[idx, ['uphill', 'downhill']] = tuple(gpx.get_uphill_downhill())
                data.loc[idx, 'gpx'] = gpx.to_xml()
                gpx_ok.loc[idx] = True
            del gpx
    del elevation_data
    
    print()
    
    # Rimozione righe senza dati di altitudine o con GPX errato
    drop_where(data, ~gpx_ok)
    
    # Difficulty categorica
    order = [
        'T1- - Valley hike',
        'T1 - Valley hike',
        'T1+ - Valley hike',
        'T2- - Mountain hike',
        'T2 - Mountain hike',
        'T2+ - Mountain hike',
        'T3- - Difficult Mountain hike',
        'T3 - Difficult Mountain hike',
        'T3+ - Difficult Mountain hike',
        'T4- - High-level Alpine hike',
        'T4 - High-level Alpine hike',
        'T4+ - High-level Alpine hike',
        'T5- - Challenging High-level Alpine hike',
        'T5 - Challenging High-level Alpine hike',
        'T5+ - Challenging High-level Alpine hike',
        'T6- - Difficult High-level Alpine hike',
        'T6 - Difficult High-level Alpine hike',
        'T6+ - Difficult High-level Alpine hike',
    ]
    data['difficulty'] = pd.Categorical(data['difficulty'], order, ordered=True)
    
    return data

def update_progress(blocks, block_size, total_size):
    bar_len = 20 # Characters
    percent = (blocks * block_size) / total_size
    progress = int(percent * bar_len)
    text = 'Downloading file [{0}] {1:.1f}%'.format('#' * progress + '-' * (bar_len - progress), percent*100)
    clear_output(wait=True)
    print(text)

def prepare_data():
    if not os.path.exists(DATA_FILE):
        if not os.path.exists(CSV_FILE):
            if not os.path.exists(ZIP_FILE):
                raise FileNotFoundError(f'Download the dataset from {URL} and save it as {ZIP_FILE}')
            print('Extracting file')
            with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
                zip_ref.extractall('.')
        raw_data = pd.read_csv(CSV_FILE)
        data = clean_data(raw_data)
        data.to_pickle(DATA_FILE)
        del data
        del raw_data
        print("Done")
        
def highlight_zero(s):
    '''
    Pandas style to highlight the zeros in a Series yellow.
    '''
    is_max = s == 0
    return ['background-color: yellow' if v else '' for v in is_max]

def highlight_where(mask, color='yellow'):
    '''
    Returns a Pandas style to highlight the cells where 'mask' is True using the specified color.
    Use `df.style.apply(highlight_where(...))`
    '''
    def style(s):
        return np.where(mask[s.name], f'background-color: {color}', '')
    return style
