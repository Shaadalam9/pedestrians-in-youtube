from flask import Flask, request, render_template_string
import pandas as pd
import os
import common
from pytube import YouTube

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
                                     'fps_list', 'geni', 'traffic_index'])


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
    gdp_city = ''
    population_city = ''
    population_country = ''
    traffic_mortality = ''
    continent = ''
    literacy_rate = ''
    avg_height = ''
    upload_date = ''
    fps_list = ''
    geni = ''
    traffic_index = ''

    if request.method == 'POST':
        if 'fetch_data' in request.form:
            city = request.form.get('city')
            country = request.form.get('country')
            state = request.form.get('state')
            video_url = request.form.get('video_url')

            # Check if city, state and country exist in the CSV
            if state:    
                existing_data = df[(df['city'] == city) & (df['country'] == country)]
            else:
                existing_data = df[(df['city'] == city) & (df['city'] == city) & (df['country'] == country)]
            if not existing_data.empty:
                existing_data_row = existing_data.iloc[0].to_dict()  # Convert to dictionary
                state = existing_data_row.get('state', '')
                video_id = existing_data_row.get('videos', '').split(',')[0] if pd.notna(existing_data_row.get('videos', '')) else ''  # noqa: E501
            else:
                message = "No existing entry found for this city (state) and country. You can add new data."

        elif 'submit_data' in request.form:
            city = request.form.get('city')
            country = request.form.get('country')
            state = request.form.get('state')
            video_url = request.form.get('video_url')
            time_of_day = request.form.getlist('time_of_day')
            start_time = request.form.getlist('start_time')
            end_time = request.form.getlist('end_time')
            gdp_city = request.form.get('gdp_city')
            population_city = request.form.get('population_city')
            population_country = request.form.get('population_country')
            traffic_mortality = request.form.get('traffic_mortality')
            continent = request.form.get('continent')
            literacy_rate = request.form.get('literacy_rate')
            avg_height = request.form.get('avg_height')
            upload_date_video = request.form.get('upload_date_video')
            fps_video = request.form.get('fps_video')
            geni = request.form.get('geni')
            traffic_index = request.form.get('traffic_index')

            # Validate Time of Day and End Time > Start Time
            if any(t not in ['0', '1'] for t in time_of_day):
                message = "Time of Day must be either 0 or 1."
            elif any(int(et) <= int(st) for st, et in zip(start_time, end_time)):
                message = "End Time must be larger than Start Time."
            else:
                # Extract ISO Country code (dummy logic here, replace with actual mapping if available)
                ISO_country = country[:3].upper()

                # Extract video ID from YouTube URL
                try:
                    yt = YouTube(video_url)
                    video_id = yt.video_id
                except Exception as e:
                    return render_template_string(
                        template, message=f"Invalid YouTube URL: {e}", df=df, city=city, country=country, state=state,
                        video_url=video_url, video_id=video_id, existing_data=existing_data_row
                    )

                # Check if the city is already present in the CSV
                if state:    
                    check_existing = city in df['city'].values and state in df['state'].values and country in df['country'].values
                else:
                    check_existing = city in df['city'].values and country in df['country'].values
                if check_existing:
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
                    upload_date_list = df.at[idx, 'upload_date'].split(',') if pd.notna(df.at[idx, 'upload_date']) else []
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
                    df.at[idx, 'gdp_city_(billion_US)'] = float(gdp_city)
                    df.at[idx, 'population_city'] = float(population_city)
                    df.at[idx, 'population_country'] = float(population_country)
                    df.at[idx, 'traffic_mortality'] = float(traffic_mortality)
                    df.at[idx, 'continent'] = continent
                    df.at[idx, 'literacy_rate'] = float(literacy_rate)
                    df.at[idx, 'avg_height'] = float(avg_height)
                    upload_date_list = [int(x) for x in upload_date_list]
                    df.at[idx, 'upload_date'] = str(upload_date_list)
                    fps_list = [float(x) for x in fps_list]
                    df.at[idx, 'fps_list'] = str(fps_list)
                    df.at[idx, 'geni'] = float(geni)
                    df.at[idx, 'traffic_index'] = float(traffic_index)

                else:
                    # Add new row if city and country are not found in the CSV
                    new_row = {
                        'city': city,
                        'state': state,
                        'country': country,
                        'ISO_country': ISO_country,
                        'videos': video_id,
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
                        'upload_date': upload_date,
                        'fps_list': fps_list,
                        'geni': geni,
                        'traffic_index': traffic_index,
                    }
                    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

                # Save to CSV
                save_csv(df, FILE_PATH)
                message = "Data added/updated successfully!"

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

    return render_template_string(
        template, message=message, df=df, city=city, country=country, state=state, video_url=video_url,
        video_id=video_id, existing_data=existing_data_row
    )


# HTML Template
template = """
<!doctype html>
<html lang="en">
<head>
    <title>Extend Mapping CSV</title>
    <style>
        .container {
            display: flex;
            justify-content: space-between;
        }
        .left, .right {
            width: 48%;
        }
        .left {
            padding: 15px;
            height: 90vh;
            overflow-y: auto;
        }
        .right {
            width: 48%;
        }
        .videos-text {
            word-wrap: break-word;
            white-space: normal;
        }
        .message {
            color: green;
            margin-top: 20px;
        }
    </style>
    <script src="https://www.youtube.com/iframe_api"></script>
    <script>
        // Declare the YouTube player variable
        let player;

        // This function will be called once the API is ready
        function onYouTubeIframeAPIReady() {
            player = new YT.Player('embeddedVideo', {
                events: {
                    'onStateChange': onPlayerStateChange
                }
            });
        }

        // This function will be triggered whenever the state of the video changes
        function onPlayerStateChange(event) {
            // Check if the video is playing
            if (event.data == YT.PlayerState.PLAYING) {
                updateCurrentTime(); // Update the time while video is playing
            }
        }

        // Function to update the current time displayed
        function updateCurrentTime() {
            // Get the current time from the YouTube player
            const currentTime = player.getCurrentTime();
            
            // Update the current time displayed on the page
            document.getElementById('currentTime').textContent = Math.floor(currentTime);
            
            // Continuously update the time every second while the video is playing
            setTimeout(updateCurrentTime, 1000);
        }
    </script>
</head>
<body>
    <h1>Extend Mapping CSV</h1>
    <div class="container">
        <div class="left">
            <form method="POST" onsubmit="return validateTimes()">
                <label for="city">City:</label>
                <input type="text" id="city" name="city" value="{{ city }}" required size="30"><br>

                <label for="state">State (optional):</label>
                <input type="text" id="state" name="state" value="{{ state }}" size="20"><br>

                <label for="country">Country:</label>
                <input type="text" id="country" name="country" value="{{ country }}" required size="30"><br>

                <label for="video_url">YouTube video URL:</label>
                <input type="url" id="video_url" name="video_url" value="{{ video_url }}" required size="50"><br>

                <button type="submit" name="fetch_data">Fetch Data</button>
            </form>

            {% if existing_data %}
            <h3>Existing data:</h3>
            <p class="videos-text">Videos: {{ existing_data['videos'] }}</p>
            <p class="videos-text">Time of day: {{ existing_data['time_of_day'] }}</p>
            <p class="videos-text">Start time: {{ existing_data['start_time'] }}</p>
            <p class="videos-text">End time: {{ existing_data['end_time'] }}</p>
            <p class="videos-text">Upload date: {{ existing_data['upload_date'] }}</p>
            <p class="videos-text">FPS list: {{ existing_data['fps_list'] }}</p>

            <!-- Form for submitting additional data -->
            <form method="POST" onsubmit="return validateTimes()">
                <input type="hidden" name="city" value="{{ city }}">
                <input type="hidden" name="country" value="{{ country }}">
                <input type="hidden" name="video_url" value="{{ video_url }}">

                <label for="time_of_day">Time of day:</label>
                <select id="time_of_day" name="time_of_day">
                    <option value="0">0 (day)</option>
                    <option value="1">1 (night)</option>
                </select><br>

                <label for="start_time">Start time (seconds):</label>
                <input type="text" id="start_time" name="start_time" required><br>

                <label for="end_time">End time (seconds):</label>
                <input type="text" id="end_time" name="end_time" required><br>

                <label for="upload_date_video">Upload date:</label>
                <input type="text" id="upload_date_video" name="upload_date_video" value="{{ upload_date_video }}"><br>

                <label for="fps_video">FPS:</label>
                <select id="fps_video" name="fps_video">
                    <option value="30.0" {% if fps_video == '30.0' %}selected{% endif %}>30.0</option>
                    <option value="60.0" {% if fps_video == '60.0' %}selected{% endif %}>60.0</option>
                </select><br>

                <!-- Fields for additional data -->
                <label for="gdp_city">GDP city (billion USD):</label>
                <input type="text" id="gdp_city" name="gdp_city" value="{{ existing_data['gdp_city_(billion_US)'] }}"><br>

                <label for="population_city">Population city:</label>
                <input type="text" id="population_city" name="population_city" value="{{ existing_data['population_city'] }}"><br>

                <label for="population_country">Population country:</label>
                <input type="text" id="population_country" name="population_country" value="{{ existing_data['population_country'] }}"><br>

                <label for="traffic_mortality">Traffic mortality:</label>
                <input type="text" id="traffic_mortality" name="traffic_mortality" value="{{ existing_data['traffic_mortality'] }}"><br>

                <label for="continent">Continent:</label>
                <select id="continent" name="continent">
                    <option value="Africa" {% if existing_data['continent'] == 'Africa' %}selected{% endif %}>Africa</option>
                    <option value="Asia" {% if existing_data['continent'] == 'Asia' %}selected{% endif %}>Asia</option>
                    <option value="Europe" {% if existing_data['continent'] == 'Europe' %}selected{% endif %}>Europe</option>
                    <option value="South America" {% if existing_data['continent'] == 'South America' %}selected{% endif %}>South America</option>
                    <option value="North America" {% if existing_data['continent'] == 'North America' %}selected{% endif %}>North America</option>
                    <option value="Oceania" {% if existing_data['continent'] == 'Oceania' %}selected{% endif %}>Oceania</option>
                </select><br>

                <label for="literacy_rate">Literacy rate:</label>
                <input type="text" id="literacy_rate" name="literacy_rate" value="{{ existing_data['literacy_rate'] }}"><br>

                <label for="avg_height">Average height:</label>
                <input type="text" id="avg_height" name="avg_height" value="{{ existing_data['avg_height'] }}"><br>

                <label for="geni">Genetic index:</label>
                <input type="text" id="geni" name="geni" value="{{ existing_data['geni'] }}"><br>

                <label for="traffic_index">Traffic index:</label>
                <input type="text" id="traffic_index" name="traffic_index" value="{{ existing_data['traffic_index'] }}"><br>

                <button type="submit" name="submit_data">Submit</button>
            </form>
            {% endif %}
        </div>

        <div class="right">
            {% if video_id %}
            <iframe id="embeddedVideo" width="100%" height="50%" src="https://www.youtube.com/embed/{{ video_id }}?enablejsapi=1" frameborder="0" allowfullscreen></iframe>
            <p>Current time: <span id="currentTime">0</span> seconds</p>
            {% endif %}
        </div>
    </div>
</body>
</html>
"""

if __name__ == "__main__":
    app.run(debug=True)
