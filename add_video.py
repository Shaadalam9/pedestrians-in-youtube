from flask import Flask, request, render_template
import pandas as pd
import os
import common
from pytube import YouTube
import webbrowser
from threading import Timer
import random
import pycountry


app = Flask(__name__)

# Get the file path using common module
FILE_PATH = common.get_configs("mapping")


def load_csv(file_path):
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        return pd.DataFrame(columns=['city', 'state', 'country', 'ISO_country', 'videos', 'time_of_day', 'start_time',
                                     'end_time', 'gdp_city_(billion_US)', 'population_city', 'population_country',
                                     'traffic_mortality', 'continent', 'literacy_rate', 'avg_height', 'upload_date',
                                     'fps_list', 'gini', 'traffic_index'])


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
    fps_list = ''
    gini = ''
    traffic_index = ''
    upload_date_video = ''
    fps_video = 0.0
    yt_title = ''
    yt_upload_date = ''
    yt_description = ''

    if request.method == 'POST':
        if 'fetch_data' in request.form:
            city = request.form.get('city')
            country = request.form.get('country')
            state = request.form.get('state')
            video_url = request.form.get('video_url')            

            # Extract video ID from YouTube URL
            try:
                yt = YouTube(video_url)
                video_url_id = yt.video_id
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
                    state=state, video_url=video_url, video_id=video_url_id, existing_data=existing_data_row,
                    fps_video=fps_video, upload_date_video=upload_date_video, yt_title=yt_title,
                    yt_description=yt_description, yt_upload_date=yt_upload_date
                )

            # Check if city, state and country exist in the CSV
            if state:    
                existing_data = df[(df['city'] == city) & (df['country'] == country)]
            else:
                existing_data = df[(df['city'] == city) & (df['city'] == city) & (df['country'] == country)]
            if not existing_data.empty:
                existing_data_row = existing_data.iloc[0].to_dict()  # Convert to dictionary
                state = existing_data_row.get('state', '')
                video_id = existing_data_row.get('videos', '').split(',')[0] if pd.notna(existing_data_row.get('videos', '')) else ''  # noqa: E501
            
                # Get the list of videos for the city and country
                videos_list = existing_data_row.get('videos', '').split(',')
                videos_list = [video.strip('[]') for video in videos_list]
                fps_list = existing_data_row.get('fps_list', '').split(',')
                fps_list = [fps.strip('[]') for fps in fps_list]
                upload_date_list = existing_data_row.get('upload_date', '').split(',')
                upload_date_list = [update_date.strip('[]') for update_date in upload_date_list]
                if video_url_id in videos_list:
                    position = videos_list.index(video_url_id)
                    fps_video = fps_list[position].strip()
                    upload_date_video = upload_date_list[position].strip()
            else:
                message = "No existing entry found for this city (state) and country. You can add new data."
                existing_data_row = {'city': city,
                                     'country': country,
                                     'ISO_country': get_iso3_country_code(country),
                                     'state': city,
                                     'videos': [],
                                     'time_of_day': [],
                                     'gdp_city_(billion_US)': 0.0,
                                     'population_city': 0.0,
                                     'population_country': 0.0,
                                     'traffic_mortality': 0.0,
                                     'start_time': [],
                                     'end_time': [],
                                     'continent': '',
                                     'literacy_rate': 0.0,
                                     'avg_height': 0.0,
                                     'upload_date': [],
                                     'fps_list': [],
                                     'gini': 0.0,
                                     'traffic_index': 0.0}
                video_id = video_url_id

        elif 'submit_data' in request.form:
            city = request.form.get('city')
            country = request.form.get('country')
            state = request.form.get('state')
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
                message = "Time of Day must be either 0 or 1."
            elif any(int(et) <= int(st) for st, et in zip(start_time, end_time)):
                message = "End Time must be larger than Start Time."
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
                    # Get the list of videos for the city and country
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

                    # Check if the video_id already exists in the list
                    if video_id not in videos_list:
                        # If the video doesn't exist, append it to the list
                        videos_list.append(video_id)
                        time_of_day_list.append([int(time_of_day[-1])])  # Append Time of Day as integer
                        start_time_list.append([int(start_time[-1])])    # Append Start Time as integer
                        end_time_list.append([int(end_time[-1])])        # Append End Time as integer
                        upload_date_list.append(int(upload_date_video))
                        fps_list.append(float(fps_video))
                    else:
                        # If the video already exists, update the corresponding lists with the new data
                        video_index = videos_list.index(video_id)  # Find the index of the existing video ID
                        time_of_day_list[video_index].append(int(time_of_day[-1]))  # Append new time of day
                        start_time_list[video_index].append(int(start_time[-1]))    # Append new start time
                        end_time_list[video_index].append(int(end_time[-1]))        # Append new end time
                        upload_date_list[video_index] = int(upload_date_video)
                        fps_list[video_index] = float(fps_video)

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
                    upload_date_list = [int(x) for x in upload_date_list]
                    df.at[idx, 'upload_date'] = str(upload_date_list)
                    fps_list = [float(x) for x in fps_list]
                    df.at[idx, 'fps_list'] = str(fps_list)
                    print(gini)
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
                        'gini': gini,
                        'traffic_index': traffic_index,
                    }
                    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

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
        timestamp=end_time_input, yt_title=yt_title, yt_description=yt_description,
        yt_upload_date=yt_upload_date
    )


def get_iso3_country_code(country_name):
    try:
        country = pycountry.countries.get(name=country_name)
        if country:
            return country.alpha_3  # ISO3 code
        else:
            return "Country not found"
    except KeyError:
        return "Country not found"


if __name__ == "__main__":
    port = 5000 + random.randint(0, 999)
    url = "http://127.0.0.1:{0}".format(port)
    Timer(1.25, lambda: webbrowser.open(url)).start()
    app.run(port=port, debug=False)
