"""Adding new data to the mapping file."""
# by Pavlo Bazilinskyy <pavlo.bazilinskyy@gmail.com>
from flask import Flask, request, render_template
import pandas as pd
import os
import common
from pytube import YouTube
import webbrowser
from threading import Timer
import random
import requests
import ast
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
from datetime import datetime


app = Flask(__name__)

FILE_PATH = common.get_configs("mapping")     # mapping file

height_data = pd.read_csv(os.path.join(common.root_dir, 'height_data.csv'))  # average height data


def load_csv(file_path):
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        return pd.DataFrame(columns=['city', 'state', 'country', 'iso3', 'videos', 'time_of_day',
                                     'vehicle_type', 'start_time', 'end_time', 'gmp',
                                     'population_city', 'population_country', 'traffic_mortality', 'continent',
                                     'literacy_rate', 'avg_height', 'upload_date', 'fps_list', 'gini',
                                     'traffic_index'])


def save_csv(df, file_path):
    df.to_csv(file_path, index=False)


@app.route('/', methods=['GET', 'POST'])
def form():
    df = load_csv(FILE_PATH)
    message = ''
    city = ''
    country = ''
    video_url = ''
    state = ''
    existing_data_row = None
    video_id = ''
    time_of_day = []
    start_time = []
    end_time = []
    end_time_input = 0
    gmp = ''
    population_city = ''
    population_country = ''
    traffic_mortality = ''
    continent = ''
    literacy_rate = ''
    avg_height = ''
    upload_date_list = ''
    fps_list = ''
    vehicle_type_list = ''
    gini = ''
    traffic_index = ''
    upload_date_video = ''
    fps_video = '30'
    yt_title = ''
    yt_upload_date = ''
    yt_description = ''
    vehicle_type_video = 0
    start_time_video = []
    end_time_video = []
    vehicle_type_video = 0

    if request.method == 'POST':
        if 'fetch_data' in request.form:
            city = request.form.get('city')
            # check for missing data
            if city == 'None' or city == 'nan':
                city = None
            elif city is not None:
                city = city.strip()
            country = request.form.get('country')
            # check for missing data
            if country == 'None' or country == 'nan':
                country = None
            elif country is not None:
                country = country.strip()
            state = request.form.get('state')
            # check for missing data
            if state == 'None' or state == 'nan':
                state = None
            elif state is not None:
                state = state.strip()
            video_url = request.form.get('video_url')

            # Extract video ID from YouTube URL
            try:
                yt = YouTube(video_url)
                video_id = yt.video_id
                # get info of video
                yt_upload_date = yt.publish_date
                for n in range(6):
                    try:
                        yt_stream = yt.streams.filter(only_audio=True).first()
                        yt_stream.download(output_path='_output')
                        yt_title = yt.title
                    except:  # noqa: E722
                        continue
                for n in range(6):
                    try:
                        yt_description = yt.initial_data["engagementPanels"][n]["engagementPanelSectionListRenderer"]["content"]["structuredDescriptionContentRenderer"]["items"][1]["expandableVideoDescriptionBodyRenderer"]["attributedDescriptionBodyText"]["content"]  # noqa: E501
                    except:  # noqa: E722
                        continue
            except Exception as e:
                return render_template(
                    "add_video.html", message=f"Invalid YouTube URL: {e}", df=df, city=city, country=country,
                    state=state, video_url=video_url, video_id=video_id, existing_data=existing_data_row,
                    fps_video=fps_video, upload_date_video=upload_date_video, yt_title=yt_title,
                    yt_description=yt_description, yt_upload_date=yt_upload_date
                )

            # Check if city, state and country exist in the CSV
            if state:
                existing_data = df[(df['city'] == city) & (df['state'] == state) & (df['country'] == country)]
            else:
                existing_data = df[(df['city'] == city) & (df['country'] == country)]
            if not existing_data.empty:
                message = "Entry for city found. You can update data."
                existing_data_row = existing_data.iloc[0].to_dict()  # Convert to dictionary

                # Get the list of videos for the city (state) and country
                videos_list = existing_data_row.get('videos', '').split(',')
                videos_list = [video.strip('[]') for video in videos_list]
                fps_list = existing_data_row.get('fps_list', '').split(',')
                fps_list = [fps.strip('[]') for fps in fps_list]
                upload_date_list = existing_data_row.get('upload_date', '').split(',')
                upload_date_list = [upload_date.strip('[]') for upload_date in upload_date_list]
                vehicle_type_list = existing_data_row.get('vehicle_type', '').split(',')
                vehicle_type_list = [vehicle_type.strip('[]') for vehicle_type in vehicle_type_list]
                if video_id in videos_list:
                    position = videos_list.index(video_id)
                    fps_video = fps_list[position].strip()
                    upload_date_video = upload_date_list[position].strip()
                    start_time_list = ast.literal_eval(existing_data_row.get('start_time', ''))
                    start_time_video = start_time_list[position]
                    end_time_list = ast.literal_eval(existing_data_row.get('end_time', ''))
                    end_time_video = end_time_list[position]
            else:
                message = "No entry for city found. You can add new data."
                iso2_code = common.get_iso2_country_code(common.correct_country(country))
                iso3_code = common.get_iso3_country_code(common.correct_country(country))
                country_data = get_country_data(iso3_code)
                city_data = get_city_data(city, iso2_code)
                if iso2_code == 'XK':
                    country_population = 1578000
                else:
                    country_population = get_country_population(country_data)
                # get coordinates
                lat, lon = get_coordinates(city, state, common.correct_country(country))
                existing_data_row = {'city': city,
                                     'country': country,
                                     'iso3': iso3_code,
                                     'lat': lat,
                                     'lon': lon,
                                     'state': state,
                                     'videos': [],
                                     'time_of_day': [],
                                     'gmp': 0.0,  # get_gmp(city, state, iso3),
                                     'population_city': get_city_population(city_data),
                                     'population_country': country_population,
                                     'traffic_mortality': get_country_traffic_mortality(iso3_code),
                                     'start_time': [],
                                     'end_time': [],
                                     'continent': get_country_continent(country_data),
                                     'literacy_rate': get_country_literacy_rate(iso3_code),
                                     'avg_height': get_country_average_height(iso3_code),
                                     'upload_date': [],
                                     'fps_list': [],
                                     'vehicle_type': [],
                                     'gini': get_country_gini(country_data),
                                     'traffic_index': get_traffic_index_lat_lon(lat, lon)}  # alternative is paid Nombeo API  # noqa: E501

        elif 'submit_data' in request.form:
            city = request.form.get('city')
            # check for missing data
            if city == 'None' or city == 'nan':
                city = None
            elif city is not None:
                city = city.strip()
            country = request.form.get('country')
            # check for missing data
            if country == 'None' or country == 'nan':
                country = None
            elif country is not None:
                country = country.strip()
            state = request.form.get('state')
            # check for missing data
            if state == 'None' or state == 'nan':
                state = None
            elif state is not None:
                state = state.strip()
            video_url = request.form.get('video_url')
            time_of_day = request.form.getlist('time_of_day')
            start_time = request.form.getlist('start_time')
            end_time = request.form.getlist('end_time')
            end_time_input = int(end_time[0])  # for starting video at the last end time
            gmp = request.form.get('gmp')
            population_city = request.form.get('population_city')
            lat = request.form.get('lat')
            lon = request.form.get('lon')
            population_country = request.form.get('population_country')
            traffic_mortality = request.form.get('traffic_mortality')
            continent = request.form.get('continent')
            literacy_rate = request.form.get('literacy_rate')
            avg_height = request.form.get('avg_height')
            upload_date_video = request.form.get('upload_date_video')
            # fps_video = request.form.get('fps_video')
            fps_video = '30'
            vehicle_type_video = request.form.get('vehicle_type')
            gini = request.form.get('gini')
            traffic_index = request.form.get('traffic_index')

            # Extract video ID from YouTube URL
            try:
                yt = YouTube(video_url)
                video_id = yt.video_id
                # get info of video
                yt_upload_date = yt.publish_date
                for n in range(6):
                    try:
                        yt_stream = yt.streams.filter(only_audio=True).first()
                        yt_stream.download(output_path='_output')
                        # getting title is not stable
                        yt_title = yt.title
                    except:  # noqa: E722
                        continue
                for n in range(6):
                    try:
                        yt_description = yt.initial_data["engagementPanels"][n]["engagementPanelSectionListRenderer"]["content"]["structuredDescriptionContentRenderer"]["items"][1]["expandableVideoDescriptionBodyRenderer"]["attributedDescriptionBodyText"]["content"]  # noqa: E501
                    except:  # noqa: E722
                        continue
            except Exception as e:
                return render_template(
                    "add_video.html", message=f"Invalid YouTube URL: {e}", df=df, city=city, country=country,
                    state=state, video_url=video_url, video_id=video_id, existing_data=existing_data_row,
                    fps_video=fps_video, upload_date_video=upload_date_video, yt_title=yt_title,
                    yt_description=yt_description, yt_upload_date=yt_upload_date
                )

            # Validate Time of Day and End Time > Start Time
            if any(t not in ['0', '1'] for t in time_of_day):
                message = "Time of day must be either 0 or 1."
            elif any(v not in ['0', '1', '2', '3', '4'] for v in vehicle_type_video):
                message = "Type of vehicle must be one of: 0, 1, 2, 3, 4."
            elif any(int(et) <= int(st) for st, et in zip(start_time, end_time)):
                message = "End time must be larger than start time."
            else:
                # Check if the city is already present in the CSV
                if state:
                    check_existing = city in df['city'].values and state in df['state'].values and country in df['country'].values  # noqa: E501
                else:
                    check_existing = city in df['city'].values and country in df['country'].values
                if check_existing:
                    if state:
                        idx = df[(df['city'] == city) & (df['state'] == state) & (df['country'] == country)].index[0]
                    else:
                        idx = df[(df['city'] == city) & (df['country'] == country)].index[0]
                    # Get the list of videos for the city (and state) and country
                    videos_list = df.at[idx, 'videos'].split(',') if pd.notna(df.at[idx, 'videos']) else []
                    # Clean the individual video IDs by stripping any leading or trailing brackets
                    videos_list = [video.strip('[]') for video in videos_list]
                    time_of_day_list = eval(df.at[idx, 'time_of_day']) if pd.notna(df.at[idx, 'time_of_day']) else []
                    start_time_list = eval(df.at[idx, 'start_time']) if pd.notna(df.at[idx, 'start_time']) else []
                    end_time_list = eval(df.at[idx, 'end_time']) if pd.notna(df.at[idx, 'end_time']) else []
                    fps_list = df.at[idx, 'fps_list'].split(',') if pd.notna(df.at[idx, 'fps_list']) else []
                    fps_list = [fps.strip('[]') for fps in fps_list]
                    upload_date_list = df.at[idx, 'upload_date'].split(',') if pd.notna(df.at[idx, 'upload_date']) else []  # noqa: E501
                    upload_date_list = [upload_date.strip('[]') for upload_date in upload_date_list]
                    vehicle_type_list = df.at[idx, 'vehicle_type'].split(',') if pd.notna(df.at[idx, 'vehicle_type']) else []  # noqa: E501
                    vehicle_type_list = [vehicle_type.strip('[]') for vehicle_type in vehicle_type_list]

                    # Check if the video_id already exists in the list
                    if video_id not in videos_list:
                        # If the video doesn't exist, append it to the list
                        videos_list.append(video_id)
                        video_index = videos_list.index(video_id)  # Find the index of the existing video ID
                        time_of_day_list.append([int(time_of_day[-1])])    # Append time of day as integer
                        start_time_list.append([int(start_time[-1])])      # Append start time as integer
                        end_time_list.append([int(end_time[-1])])          # Append end time as integer
                        upload_date_list.append(int(upload_date_video))    # Append upload time as integer
                        fps_list.append(int(fps_video))                    # Append fps list as integer
                        vehicle_type_list.append(int(vehicle_type_video))  # Append vehicle type as integer
                    else:
                        # If the video already exists, update the corresponding lists with the new data
                        video_index = videos_list.index(video_id)  # Find the index of the existing video ID
                        time_of_day_list[video_index].append(int(time_of_day[-1]))  # Append new time of day
                        start_time_list[video_index].append(int(start_time[-1]))    # Append new start time
                        end_time_list[video_index].append(int(end_time[-1]))        # Append new end time
                        if upload_date_video != 'None':
                            upload_date_list[video_index] = int(upload_date_video)
                        else:
                            upload_date_list[video_index] = upload_date_video
                        fps_list[video_index] = float(fps_video)
                        vehicle_type_list[video_index] = int(vehicle_type_video)
                    start_time_video = start_time_list[video_index]
                    end_time_video = end_time_list[video_index]

                    # Update the DataFrame row with the modified lists and new data
                    df.at[idx, 'videos'] = '[' + ','.join(videos_list) + ']'  # Join the list as a string
                    df.at[idx, 'time_of_day'] = str(time_of_day_list)  # Store as string representation
                    df.at[idx, 'start_time'] = str(start_time_list)    # Store as string representation
                    df.at[idx, 'end_time'] = str(end_time_list)        # Store as string representation
                    if gmp:
                        df.at[idx, 'gmp'] = float(gmp)
                    else:
                        df.at[idx, 'gmp'] = 0.0
                    if population_city:
                        df.at[idx, 'population_city'] = float(population_city)
                    else:
                        df.at[idx, 'population_city'] = 0.0
                    if population_country:
                        df.at[idx, 'population_country'] = float(population_country)
                    else:
                        df.at[idx, 'population_country'] = 0.0
                    if traffic_mortality:
                        df.at[idx, 'traffic_mortality'] = float(traffic_mortality)
                    else:
                        df.at[idx, 'traffic_mortality'] = 0.0
                    df.at[idx, 'continent'] = continent
                    if literacy_rate:
                        df.at[idx, 'literacy_rate'] = float(literacy_rate)
                    else:
                        df.at[idx, 'literacy_rate'] = 0.0
                    if avg_height:
                        df.at[idx, 'avg_height'] = float(avg_height)
                    else:
                        df.at[idx, 'avg_height'] = 0.0
                    if lat:
                        df.at[idx, 'lat'] = float(lat)
                    else:
                        df.at[idx, 'lat'] = 0.0
                    if lon:
                        df.at[idx, 'lon'] = float(lon)
                    else:
                        df.at[idx, 'lon'] = 0.0
                    for i in range(len(upload_date_list)):
                        if upload_date_list[i] != 'None':
                            upload_date_list[i] = int(upload_date_list[i])
                    upload_date_list = str(upload_date_list)
                    upload_date_list = upload_date_list.replace('\'', '')
                    upload_date_list = upload_date_list.replace(' ', '')
                    df.at[idx, 'upload_date'] = upload_date_list
                    fps_list = [30.0 if str(x).strip().lower() == 'none' else float(x) for x in fps_list]
                    fps_list = str(fps_list)
                    fps_list = fps_list.replace('\'', '')
                    fps_list = fps_list.replace(' ', '')
                    print(fps_list)
                    df.at[idx, 'fps_list'] = fps_list
                    vehicle_type_list = [int(x) for x in vehicle_type_list]
                    vehicle_type_list = str(vehicle_type_list)
                    vehicle_type_list = vehicle_type_list.replace('\'', '')
                    vehicle_type_list = vehicle_type_list.replace(' ', '')
                    df.at[idx, 'vehicle_type'] = vehicle_type_list
                    if gini:
                        df.at[idx, 'gini'] = float(gini)
                    else:
                        df.at[idx, 'gini'] = 0.0
                    if traffic_index:
                        df.at[idx, 'traffic_index'] = float(traffic_index)
                    else:
                        df.at[idx, 'traffic_index'] = 0.0

                else:
                    # Add new row if city and country are not found in the CSV
                    new_row = {
                        'id': int(df.iloc[-1]['id']+1),
                        'city': city,
                        'state': state,
                        'country': country,
                        'iso3': common.get_iso3_country_code(common.correct_country(country)),
                        'lat': lat,
                        'lon': lon,
                        'videos': '[' + video_id + ']',
                        'time_of_day': str([[int(x) for x in time_of_day]]),  # Store as stringified list of integers
                        'start_time': str([[int(x) for x in start_time]]),    # Store as stringified list of integers
                        'end_time': str([[int(x) for x in end_time]]),        # Store as stringified list of integers
                        'gmp': gmp,
                        'population_city': population_city,
                        'population_country': population_country,
                        'traffic_mortality': traffic_mortality,
                        'continent': continent,
                        'literacy_rate': literacy_rate,
                        'avg_height': avg_height,
                        'upload_date': '[' + upload_date_video.strip() + ']',
                        'fps_list': '[' + fps_video.strip() + ']',
                        'vehicle_type': '[' + vehicle_type_video.strip() + ']',
                        'gini': gini,
                        'traffic_index': traffic_index,
                    }
                    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                    start_time_video = [int(x) for x in start_time]
                    end_time_video = [int(x) for x in end_time]

                # Save to CSV
                save_csv(df, FILE_PATH)
                message = "Video added/updated successfully."

    # Fetch the existing data after submit to display for the city
    if state:
        if city and state and country:
            existing_data = df[(df['city'] == city) & (df['state'] == state) & (df['country'] == country)]
            if not existing_data.empty:
                existing_data_row = existing_data.iloc[0].to_dict()  # Convert to dictionary
    else:
        if city and country:
            existing_data = df[(df['city'] == city) & (df['country'] == country)]
            if not existing_data.empty:
                existing_data_row = existing_data.iloc[0].to_dict()  # Convert to dictionary

    if not upload_date_video and yt_upload_date:
        upload_date_video = yt_upload_date.strftime('%d%m%Y')
    return render_template(
        "add_video.html", message=message, df=df, city=city, country=country, state=state, video_url=video_url,
        video_id=video_id, existing_data=existing_data_row, fps_video=fps_video, upload_date_video=upload_date_video,
        timestamp=end_time_input, yt_title=yt_title, yt_description=yt_description, yt_upload_date=yt_upload_date,
        start_time_video=start_time_video, end_time_video=end_time_video
    )


# Fetch country data based on its ISO-3 code
def get_country_data(iso3_code):
    try:
        # REST Countries API URL
        api_url = f"https://restcountries.com/v3.1/alpha/{iso3_code}"
        response = requests.get(api_url)
        if response.status_code == 200:
            country_data = response.json()
            # return country data
            return country_data
        else:
            return None
    except Exception as e:
        print(f"Error fetching country data: {e}")
        return None


# Fetch continents (first in the list) by ISO-3 code
def get_country_continent(country_data):
    if country_data:
        return country_data[0]['continents'][0]
    else:
        return ''


# Fetch population by ISO-3 code
def get_country_population(country_data):
    if country_data:
        return country_data[0]['population']
    else:
        return 0.0


# Fetch population by ISO-3 code
def get_country_gini(country_data):
    if country_data:
        # Extract Gini coefficient (use first key in Gini dict if available)
        gini_data = country_data[0].get('gini', {})
        if gini_data:
            return list(gini_data.values())[0]  # Use the first available Gini value
        else:
            return 0.0
    else:
        return 0.0


# Fetch literacy rate by ISO-3 code
def get_country_literacy_rate(iso3_code):
    try:
        # World Bank API URL for literacy rate
        api_url = f"http://api.worldbank.org/v2/country/{iso3_code}/indicator/SE.ADT.LITR.ZS?format=json"
        response = requests.get(api_url)
        if response.status_code == 200:
            data = response.json()
            if data and len(data) > 1 and data[1]:
                # Extract the most recent literacy rate
                for entry in data[1]:
                    if entry['value'] is not None:
                        return round(entry['value'], 2)
            return 0.0
        else:
            return 0.0
    except Exception as e:
        print(f"Error fetching literacy rate: {e}")
        return 0.0


# Function to fetch traffic mortality data
def get_country_traffic_mortality(iso3_code):
    try:
        # World Bank API URL for literacy rate
        api_url = f"http://api.worldbank.org/v2/country/{iso3_code}/indicator/SH.STA.TRAF.P5?format=json"
        response = requests.get(api_url)
        if response.status_code == 200:
            data = response.json()
            if data and len(data) > 1 and data[1]:
                # Extract the most recent literacy rate
                for entry in data[1]:
                    if entry['value'] is not None:
                        return round(entry['value'], 2)
            return 0.0
    except Exception as e:
        print(f"Error fetching traffic mortality rate: {e}")
        return 0.0


def get_city_data(city, country_code):
    """
    Get economic or city-related data from the Geonames API

    :param city: Name of the city
    :param country_code: 2-letter ISO code of the country
    :return: City data
    """
    url = f"http://api.geonames.org/searchJSON?q={city}&country={country_code}&username={common.get_secrets('geonames_username')}"  # noqa: E501
    response = requests.get(url)

    if response.status_code == 200:
        return response.json()
    else:
        return None


def get_city_population(city_data):
    """
    Get economic or city-related data from the Geonames API

    :param city: Name of the city
    :param country_code: 2-letter ISO code of the country
    :return: City data
    """
    if 'geonames' in city_data and len(city_data['geonames']) > 0:
        return city_data['geonames'][0].get('population', None)
    else:
        return 0.0


# Fetch average height by ISO-3 code
def get_country_average_height(iso3_code):
    try:
        # Filter the dataset by country name
        row = height_data[height_data['cca3'].str.lower() == iso3_code.lower()]
        if not row.empty:
            # Return average male and female height
            male_height = row.iloc[0]['meanHeightMale']
            female_height = row.iloc[0]['meanHeightFemale']
            avg_height = (male_height + female_height) / 2
            return round(avg_height)  # return rounded average height
        else:
            return 0.0
    except Exception as e:
        print(f"Error fetching height data: {e}")
        return 0.0


def get_gmp(city: str, state: str, iso3: str) -> float:
    """
    Fetches Gross Metropolitan Product (GMP) for a given city, state, and ISO3 country code.

    Args:
        city (str): The city's name.
        state (str): The state (for U.S. cities).
        iso3 (str): The 3-letter ISO country code.

    Returns:
        float: The city's Gross Metropolitan Product (GMP) in USD, or None if not found.
    """
    # todo: finish with taking state into account and caching received objects (slow API)
    if iso3.upper() == "USA":
        # Use BEA API for U.S. metro areas
        url = "https://apps.bea.gov/api/data/"
        params = {
            "UserID": common.get_secrets('bea_api_key'),
            "method": "GetData",
            "datasetname": "Regional",
            "TableName": "CAGDP2",  # GDP for metro areas
            "LineCode": "1",  # Total GDP
            "GeoFIPS": "MSA",  # Metropolitan Statistical Areas
            "Year": "2022",
            "ResultFormat": "json"
        }
        response = requests.get(url, params=params)
        data = response.json()

        # Extract GMP for the given city
        for entry in data.get("BEAAPI", {}).get("Results", {}).get("Data", []):
            if city.lower() in entry["GeoName"].lower():
                return float(entry["DataValue"])

    else:
        # Use OECD API for international cities
        url = f"https://stats.oecd.org/SDMX-JSON/data/CITIES/GDP.METRO.{iso3.upper()}?json-lang=en"
        response = requests.get(url)
        data = response.json()

        # Extract GMP for the given city
        for key, value in data.get("dataSets", [{}])[0].get("observations", {}).items():
            if city.lower() in key.lower():
                return float(value[0])

    return None  # Return None if no data is found


def get_traffic_index_lat_lon(lat, lon, api="tomtom"):
    if api == "tomtom":
        url = f"https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json?key={common.get_secrets('tomtom_api_key')}&point={lat},{lon}"  # noqa: E501
        response = requests.get(url)

        if response.status_code == 200:
            data = response.json()
            if "flowSegmentData" in data:
                current_speed = data["flowSegmentData"]["currentSpeed"]
                free_flow_speed = data["flowSegmentData"]["freeFlowSpeed"]
                traffic_index = round((1 - current_speed / free_flow_speed) * 100, 2)
                return traffic_index
            else:
                return 0.0
        else:
            print(f"Error fetching traffic index for {lat}, {lon}: {response.status_code}")
            return 0.0
    elif api == "trafiklab":
        url = f"https://api.trafiklab.se/v1/trafficindex?lat={lat}&lon={lon}&apikey={common.get_secrets('trafiklab_api_key')}"  # noqa: E501

        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise an exception for HTTP errors
            data = response.json()

            # Extract traffic index or other relevant data
            traffic_index = data.get('trafficIndex', None)

            if traffic_index is not None:
                return traffic_index
            else:
                return 0.0

        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")
            return 0.0
    else:
        print(f"Wrong type of API provided {api}.")
        return 0.0


def get_traffic_index(city, state, country):
    # API endpoint for Numbeo Traffic Index with city, state, and country
    if state:
        url = f"https://www.numbeo.com/api/traffic?api_key={common.get_secrets('numbeo_api_key')}&city={city}&state={state}&country={country}"  # noqa: E501
    else:
        url = f"https://www.numbeo.com/api/traffic?api_key={common.get_secrets('numbeo_api_key')}&city={city}&country={country}"  # noqa: E501
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        data = response.json()

        # Check if the data contains traffic information
        if 'traffic_index' in data:
            traffic_index = data['traffic_index']
            return traffic_index
        else:
            print(f"No traffic index data available fro city city={city}, state={state}, country={country}.")
            return 0.0

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return 0.0


def get_coordinates(city, state, country):
    """Get city coordinates either from the pickle file or geocode them."""
    # Generate a unique user agent with the current date and time
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    user_agent = f"my_geocoding_script_{current_time}"

    # Create a geolocator with the dynamically generated user_agent
    geolocator = Nominatim(user_agent=user_agent)

    try:
        # Attempt to geocode the city and country with a longer timeout
        if state and str(state).lower() != 'nan':
            location_query = f"{city}, {state}, {country}"  # Combine city, state and country
        else:
            location_query = f"{city}, {country}"  # Combine city and country
        location = geolocator.geocode(location_query, timeout=2)  # type: ignore # Set a 2-second timeout

        if location:
            return location.latitude, location.longitude  # type: ignore
        else:
            print(f"Failed to geocode {location_query}")
            return None, None  # Return None if city is not found

    except GeocoderTimedOut:
        print(f"Geocoding timed out for {location_query}.")
    except GeocoderUnavailable:
        print(f"Geocoding server could not be reached for {location_query}.")
        return None, None  # Return None if city is not found


if __name__ == "__main__":
    port = 5000 + random.randint(0, 999)
    url = "http://127.0.0.1:{0}".format(port)
    Timer(1.25, lambda: webbrowser.open(url)).start()
    app.run(port=port, debug=False)
