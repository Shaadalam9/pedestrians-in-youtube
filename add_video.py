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
import pycountry
import requests
import ast


app = Flask(__name__)


FILE_PATH = common.get_configs("mapping")     # mapping file

height_data = pd.read_csv(os.path.join(common.root_dir, 'height_data.csv'))  # average height data


def load_csv(file_path):
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        return pd.DataFrame(columns=['city', 'state', 'country', 'ISO_country', 'videos', 'time_of_day',
                                     'vehicle_type', 'start_time', 'end_time', 'gdp_city_(billion_US)',
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
    gdp_city = ''
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
    fps_video = 0
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
            country = request.form.get('country')
            # check for missing data
            if country == 'None' or country == 'nan':
                country = None
            state = request.form.get('state')
            # check for missing data
            if state == 'None' or state == 'nan':
                state = None
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
                existing_data_row = existing_data.iloc[0].to_dict()  # Convert to dictionary
                state = existing_data_row.get('state', '')

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
                message = "No existing entry found. You can add new data."
                iso2_code = get_iso2_country_code(country)
                iso3_code = get_iso3_country_code(country)
                country_data = get_country_data(iso3_code)
                city_data = get_city_data(city, iso2_code)
                existing_data_row = {'city': city,
                                     'country': country,
                                     'ISO_country': iso3_code,
                                     'state': city,
                                     'videos': [],
                                     'time_of_day': [],
                                     'gdp_city_(billion_US)': 0.0,
                                     'population_city': get_city_population(city_data),
                                     'population_country': get_country_population(country_data),
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
                                     'traffic_index': 0.0}

        elif 'submit_data' in request.form:
            city = request.form.get('city')
            # check for missing data
            if city == 'None' or city == 'nan':
                city = None
            country = request.form.get('country')
            # check for missing data
            if country == 'None' or country == 'nan':
                country = None
            state = request.form.get('state')
            # check for missing data
            if state == 'None' or state == 'nan':
                state = None
            video_url = request.form.get('video_url')
            time_of_day = request.form.getlist('time_of_day')
            start_time = request.form.getlist('start_time')
            end_time = request.form.getlist('end_time')
            end_time_input = int(end_time[0])  # for starting video at the last end time
            gdp_city = request.form.get('gdp_city')
            population_city = request.form.get('population_city')
            population_country = request.form.get('population_country')
            traffic_mortality = request.form.get('traffic_mortality')
            continent = request.form.get('continent')
            literacy_rate = request.form.get('literacy_rate')
            avg_height = request.form.get('avg_height')
            upload_date_video = request.form.get('upload_date_video')
            fps_video = request.form.get('fps_video')
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
                        fps_list.append(int(fps_video))                  # Append fps list as integer
                        vehicle_type_list.append(int(vehicle_type_video))  # Append fps list as integer
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
                    if gdp_city:
                        df.at[idx, 'gdp_city_(billion_US)'] = float(gdp_city)
                    else:
                        df.at[idx, 'gdp_city_(billion_US)'] = 0.0
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
                    for i in range(len(upload_date_list)):
                        if upload_date_list[i] != 'None':
                            upload_date_list[i] = int(upload_date_list[i])
                    upload_date_list = str(upload_date_list)
                    upload_date_list = upload_date_list.replace('\'', '')
                    upload_date_list = upload_date_list.replace(' ', '')
                    df.at[idx, 'upload_date'] = upload_date_list
                    fps_list = [float(x) for x in fps_list]
                    fps_list = str(fps_list)
                    fps_list = fps_list.replace('\'', '')
                    fps_list = fps_list.replace(' ', '')
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
                        'ISO_country': get_iso3_country_code(country),
                        'videos': '[' + video_id + ']',
                        'time_of_day': str([[int(x) for x in time_of_day]]),  # Store as stringified list of integers
                        'start_time': str([[int(x) for x in start_time]]),    # Store as stringified list of integers
                        'end_time': str([[int(x) for x in end_time]]),        # Store as stringified list of integers
                        'gdp_city_(billion_US)': gdp_city,
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


# Fetch ISO-3 country data
def get_iso3_country_code(country_name):
    try:
        if country_name == 'Russia':
            country_name = 'Russian Federation'
        country = pycountry.countries.get(name=country_name)
        if country:
            if country == 'Kosovo':
                return 'XKX'
            else:
                return country.alpha_3  # ISO-3 code
        else:
            return "Country not found"
    except KeyError:
        return "Country not found"


# Fetch ISO-2 country data
def get_iso2_country_code(country_name):
    try:
        if country_name == 'Russia':
            country_name = 'Russian Federation'
        country = pycountry.countries.get(name=country_name)
        if country:
            if country == 'Kosovo':
                return 'XK'
            elif country == 'Russia' or country == 'Russian Federation':
                return 'RU'
            else:
                return country.alpha_2  # ISO-2 code
        else:
            return "Country not found"
    except KeyError:
        return "Country not found"


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


def get_city_data(city_name, country_code):
    """
    Get economic or city-related data from the Geonames API

    :param city_name: Name of the city
    :param country_code: 2-letter ISO code of the country
    :return: City data
    """
    url = f"http://api.geonames.org/searchJSON?q={city_name}&country={country_code}&username={common.get_secrets('geonames_username')}"  # noqa: E501
    response = requests.get(url)

    if response.status_code == 200:
        return response.json()
    else:
        return None


def get_city_population(city_data):
    """
    Get economic or city-related data from the Geonames API

    :param city_name: Name of the city
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


if __name__ == "__main__":
    port = 5000 + random.randint(0, 999)
    url = "http://127.0.0.1:{0}".format(port)
    Timer(1.25, lambda: webbrowser.open(url)).start()
    app.run(port=port, debug=False)
