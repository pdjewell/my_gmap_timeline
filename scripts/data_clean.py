import os
import json
import glob
import numpy as np
import pandas as pd
import datetime as dt 
import regex as re

# Change cwd
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
#os.chdir('..')


def read_in_data(folder_path):

    # Read in data and save in list 
    files = glob.glob(folder_path + '**/*.json', 
                   recursive = True)
    data_list = []
    for f in files:
        with open(f, 'r') as file:
            data = json.load(file)
        data = data['timelineObjects']
        data_list.extend(data)
        print(f'Successfully read in this file: {f}') 

    print(f'Data is {type(data_list)} of length: {len(data_list)}')
    return data_list 


def clean_visits(data='data', home='A2215', work='St Helier Hospital'):
    # Take only place_visit data
    place_visits = [segment.get('placeVisit') for segment in data if segment.get('placeVisit') is not None]
    # Reduce data to only what is needed
    keys_needed = ['location','placeId','duration', 
                   'placeConfidence', 'visitConfidence', 
                   'locationConfidence', 'placeVisitType', 
                   'placeVisitImportance']
    place_visits = [{k : v for k, v in visit.items() 
                     if str(k) in keys_needed} 
                     for visit in place_visits]
    # Take only necessary data
    visits = []
    for visit in place_visits:
        new_dict = {}
        try:
            new_dict['location address'] = visit.get('location')['address']
        except:
            new_dict['location address'] = 'unknown'
        try:
            new_dict['location name'] = visit.get('location')['name']
        except:
            new_dict['location name'] = visit.get('location')['address'].split(',')[0]
        try:
            new_dict['placeId'] = visit.get('location')['placeId'] 
        except:
            continue
        try:
            new_dict['location latitude'] = visit.get('location')['latitudeE7'] / 1e7
            new_dict['location longitude'] = visit.get('location')['longitudeE7'] / 1e7
        except:
            pass
        new_dict['location confidence'] = visit.get('location')['locationConfidence']
        new_dict['start timestamp'] = visit.get('duration')['startTimestamp']
        new_dict['end timestamp'] = visit.get('duration')['endTimestamp']
        new_dict['visit confidence'] = visit['visitConfidence']
        new_dict['location confidence'] = visit['locationConfidence']
        new_dict['place visit importance'] = visit['placeVisitImportance'].lower()
        visits.append(new_dict)

    # Create a data frame 
    visits = pd.DataFrame(visits)

    # Create activity type column
    visits['activity type'] = 'place visit'
    # Create visit duration columns 
    visits['start timestamp'] = pd.to_datetime(visits['start timestamp'],format='ISO8601')
    visits['end timestamp'] = pd.to_datetime(visits['end timestamp'],format='ISO8601')
    visits['visit duration'] = visits['end timestamp'] - visits['start timestamp']
    visits['visit duration (in minutes)'] = (visits['visit duration'].dt.total_seconds() / 60).round(1)
    # Create clearer start and end date and time columns 
    visits['visit start date'] = visits['start timestamp'].dt.date.astype(str)
    visits['visit start time'] = visits['start timestamp'].dt.round(freq='T').dt.time
    visits['visit start day of week'] = visits['start timestamp'].dt.day_name()
    visits['visit start month'] = visits['start timestamp'].dt.month_name() 
    visits['visit start year'] = visits['start timestamp'].dt.year.astype(int)
    visits['visit end date'] = visits['end timestamp'].dt.date.astype(str)
    visits['visit end time'] = visits['end timestamp'].dt.round(freq='T').dt.time
    # Create home and work columns 
    visits['Work'] = visits['location name'].apply(lambda x: 'yes' if x == work else 'no')
    visits['Home'] = visits['location name'].apply(lambda x: 'yes' if x == home else 'no')
    # Change home location name
    visits['location name'] = visits['location name'].apply(lambda x: 'Home' if x==home else x)
    
    # Create country column that is cleaned 
    visits['country'] = visits.loc[:, 'location address'].apply(lambda x: x.split(', ')[-1])
    mapping_dict = {
        r'(^Espanya$|^España$)': 'Spain',
        r'(^UK$|^United Kingdom$|^Y Deyrnas Unedig$|^Royaume-Uni$|^Великобритания$)': 'United Kingdom',
        r'(^USA$|^United States$)': 'United States',
        r'(^New Zealand$|^Aotearoa$)': 'New Zealand',
        r'(^Italia$|^Italie$)': 'Italy',
        r'(^México$)': 'Mexico',
        r'(^Deutschland$)': 'Germany',
        r'^Singapore \d+': 'Singapore',
        r'(^Türkiye$)': 'Turkey',
        r'(^Perú$|^Peru$|^Perù$)': 'Peru',
        r'(^Kolumbien$)': 'Colombia',
        r'(^Bolivie$)': 'Bolivia',
        r'(^Ελλάδα$|^Hellas$)': 'Greece',
        r'(^Panamá$)': 'Panama',
        r'(^Nederland$)': 'Netherlands',
        r'(^België$)': 'Belgium',
        r'(^\d+ Magyarország$)': 'Hungary',
        r'(^Österreich$)': 'Austria'
    }
    visits['country'] = visits['country'].replace(mapping_dict, regex=True)

    # Reorder columns to sensible order
    new_cols = ['activity type','place visit importance','placeId',
                    'location name','Home','Work','country','location latitude', 'location longitude', 'location address',
                    'start timestamp','end timestamp','visit start year','visit start month','visit start day of week', 
                    'visit start date','visit start time', 'visit end date', 'visit end time',
                    'visit duration', 'visit duration (in minutes)', 
                    'location confidence','visit confidence']
    visits = visits.loc[:, new_cols]    
    
    print(f'Created visits DataFrame with shape: {visits.shape}')
    return visits


def clean_journeys(data='data'):
    # Take only place_visit data
    activity_segments = [segment.get('activitySegment') for segment in data 
                         if segment.get('activitySegment') is not None]
    # Reduce data to only what is needed
    keys_needed = ['startLocation','endLocation','duration',
                   'distance','activityType','confidence'] 
    activity_segments = [{k : v for k, v in segment.items() 
                          if str(k) in keys_needed} 
                          for segment in activity_segments]
    # Take only necessary data
    journeys = []
    for segment in activity_segments:
        new_dict = {}
        new_dict['journey start location latitude'] = segment.get('startLocation')['latitudeE7'] / 1e7
        new_dict['journey start location longitude'] = segment.get('startLocation')['longitudeE7'] / 1e7
        new_dict['journey end location latitude'] = segment.get('endLocation')['latitudeE7'] / 1e7
        new_dict['journey end location longitude'] = segment.get('endLocation')['longitudeE7'] / 1e7
        new_dict['start timestamp'] = segment.get('duration')['startTimestamp'] 
        new_dict['end timestamp'] = segment.get('duration')['endTimestamp'] 
        try:
            new_dict['journey distance (meters)'] = segment['distance'] 
        except:
            new_dict['journey distance (meters)'] = np.NaN
        new_dict['journey transport activity mode type'] = segment['activityType'].lower()
        journeys.append(new_dict)

    # Create a data frame 
    journeys = pd.DataFrame(journeys) 

    # Clean up activity type column
    map_dict = {'in_bus': 'bus', 
        'walking': 'walking',
        'in_passenger_vehicle': 'car',
        'in_subway': 'subway',
        'cycling': 'cycling',
        'in_train': 'train',
        'running': 'running',
        'unknown_activity_type': 'unknown',
        'flying': 'flying',
        'motorcycling':'motorcycling',
        'in_ferry':'ferry',
        'in_tram':'tram',
        'boating':'boating'}
    journeys['journey transport activity mode type'] = journeys['journey transport activity mode type'].map(map_dict)

    # Create activity type column
    journeys['activity type'] = 'journey activity segment'
    # Create journey duration columns 
    journeys['start timestamp'] = pd.to_datetime(journeys['start timestamp'],format='ISO8601')
    journeys['end timestamp'] = pd.to_datetime(journeys['end timestamp'],format='ISO8601')
    journeys['journey duration'] = journeys['end timestamp'] - journeys['start timestamp']
    journeys['journey duration (in minutes)'] = (journeys['journey duration'].dt.total_seconds() / 60).round(1)
    # Create clearer start and end date and time columns 
    journeys['journey start date'] = journeys['start timestamp'].dt.date
    journeys['journey start day of week'] = journeys['start timestamp'].dt.day_name()
    journeys['journey start month'] = journeys['start timestamp'].dt.month_name() 
    journeys['journey start year'] = journeys['start timestamp'].dt.year.astype(int)
    journeys['journey start time'] = journeys['start timestamp'].dt.round(freq='T').dt.time
    journeys['journey end date'] = journeys['end timestamp'].dt.date
    journeys['journey end time'] = journeys['end timestamp'].dt.round(freq='T').dt.time

    # Reorder columns to sensible order
    new_cols = ['activity type', 'journey transport activity mode type','journey distance (meters)',
            'journey start location latitude', 'journey start location longitude',
            'journey end location latitude', 'journey end location longitude',
            'start timestamp', 'end timestamp','journey start date', 'journey start time', 
            'journey start year', 'journey start month', 'journey start day of week',
            'journey end date','journey end time',
            'journey duration', 'journey duration (in minutes)']
    journeys = journeys.loc[:, new_cols]

    # Remove if missing journey duration and unknown activity mode type
    journeys = journeys.loc[~(journeys['journey distance (meters)'].isnull())
             | ~(journeys['journey transport activity mode type'] == 'unknown') ,:]

    print(f'Created journeys DataFrame with shape: {journeys.shape}')
    return journeys

class PlaceVisit():
    pass


if __name__ == '__main__':
    folder_path = 'location_history/'
    data = read_in_data(folder_path)
    visits = clean_visits(data)
    journeys = clean_journeys(data)
