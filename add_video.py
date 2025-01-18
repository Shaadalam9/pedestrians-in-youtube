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
        return pd.DataFrame(columns=['city', 'state', 'country', 'ISO_country', 'videos', 'time_of_day', 'start_time', 'end_time', 'gdp_city_(billion_US)', 'population_city', 'population_country', 'traffic_mortality', 'continent', 'literacy_rate', 'avg_height', 'upload_date', 'fps_list', 'geni', 'traffic_index'])

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
                video_id = existing_data_row.get('videos', '').split(',')[0] if pd.notna(existing_data_row.get('videos', '')) else ''
            else:
                message = "No existing entry found for this city and country. You can add new data."

        elif 'submit_data' in request.form:
            city = request.form.get('city')
            country = request.form.get('country')
            state = request.form.get('state')
            video_url = request.form.get('video_url')
            time_of_day = request.form.getlist('time_of_day')
            start_time = request.form.getlist('start_time')
            end_time = request.form.getlist('end_time')

            # New fields for additional data
            gdp_city = request.form.get('gdp_city')
            population_city = request.form.get('population_city')
            population_country = request.form.get('population_country')
            traffic_mortality = request.form.get('traffic_mortality')
            continent = request.form.get('continent')
            literacy_rate = request.form.get('literacy_rate')
            avg_height = request.form.get('avg_height')
            upload_date = request.form.get('upload_date')
            fps_list = request.form.get('fps_list')
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
                        template, message=f"Invalid YouTube URL: {e}", df=df, city=city, country=country, state=state, video_url=video_url, video_id=video_id, existing_data=existing_data_row
                    )

                # Check if the city is already present in the CSV
                if city in df['city'].values and country in df['country'].values:
                    idx = df[(df['city'] == city) & (df['country'] == country)].index[0]
                    # Get the list of videos for the city and country
                    videos_list = df.at[idx, 'videos'].split(',') if pd.notna(df.at[idx, 'videos']) else []
                    # Clean the individual video IDs by stripping any leading or trailing brackets
                    videos_list = [video.strip('[]') for video in videos_list]
                    time_of_day_list = eval(df.at[idx, 'time_of_day']) if pd.notna(df.at[idx, 'time_of_day']) else []
                    start_time_list = eval(df.at[idx, 'start_time']) if pd.notna(df.at[idx, 'start_time']) else []
                    end_time_list = eval(df.at[idx, 'end_time']) if pd.notna(df.at[idx, 'end_time']) else []
                    
                    # Check if the video_id already exists in the list
                    if video_id not in videos_list:
                        # If the video doesn't exist, append it to the list
                        videos_list.append(video_id)
                        time_of_day_list.append([int(time_of_day[-1])])  # Append Time of Day as integer
                        start_time_list.append([int(start_time[-1])])    # Append Start Time as integer
                        end_time_list.append([int(end_time[-1])])        # Append End Time as integer
                    else:
                        # If the video already exists, update the corresponding lists with the new data
                        video_index = videos_list.index(video_id)  # Find the index of the existing video ID
                        time_of_day_list[video_index].append(int(time_of_day[-1]))  # Append new time of day
                        start_time_list[video_index].append(int(start_time[-1]))    # Append new start time
                        end_time_list[video_index].append(int(end_time[-1]))        # Append new end time

                    # Update the DataFrame row with the modified lists and new data
                    df.at[idx, 'videos'] = ','.join(videos_list)  # Join the list as a string
                    df.at[idx, 'time_of_day'] = str(time_of_day_list)  # Store as string representation
                    df.at[idx, 'start_time'] = str(start_time_list)    # Store as string representation
                    df.at[idx, 'end_time'] = str(end_time_list)        # Store as string representation

                    # Update new columns
                    df.at[idx, 'gdp_city_(billion_US)'] = gdp_city
                    df.at[idx, 'population_city'] = population_city
                    df.at[idx, 'population_country'] = population_country
                    df.at[idx, 'traffic_mortality'] = traffic_mortality
                    df.at[idx, 'continent'] = continent
                    df.at[idx, 'literacy_rate'] = literacy_rate
                    df.at[idx, 'avg_height'] = avg_height
                    df.at[idx, 'upload_date'] = upload_date
                    df.at[idx, 'fps_list'] = fps_list
                    df.at[idx, 'geni'] = geni
                    df.at[idx, 'traffic_index'] = traffic_index

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
    if city and country:
        existing_data = df[(df['city'] == city) & (df['country'] == country)]
        if not existing_data.empty:
            existing_data_row = existing_data.iloc[0].to_dict()  # Convert to dictionary

    return render_template_string(
        template, message=message, df=df, city=city, country=country, state=state, video_url=video_url, video_id=video_id, existing_data=existing_data_row
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
    </style>
    <script>
        function updateVideoTime() {
            const video = document.getElementById('embeddedVideo');
            const currentTime = document.getElementById('currentTime');
            video.addEventListener('timeupdate', () => {
                currentTime.textContent = Math.floor(video.currentTime);
            });
        }

        function validateTimes() {
            const startTime = document.getElementById('start_time').value;
            const endTime = document.getElementById('end_time').value;

            if (parseInt(endTime) <= parseInt(startTime)) {
                alert("End Time must be larger than Start Time.");
                return false;  // Prevent form submission
            }
            return true;  // Allow form submission
        }
    </script>
</head>
<body onload="updateVideoTime()">
    <h1>Extend Mapping CSV</h1>
    <div class="container">
        <div class="left">
            <form method="POST" onsubmit="return validateTimes()">
                <label for="city">City:</label>
                <input type="text" id="city" name="city" value="{{ city }}" required><br>

                <label for="state">State (optional):</label>
                <input type="text" id="state" name="state" value="{{ state }}"><br>

                <label for="country">Country:</label>
                <input type="text" id="country" name="country" value="{{ country }}" required><br>

                <label for="video_url">YouTube video URL:</label>
                <input type="url" id="video_url" name="video_url" value="{{ video_url }}" required><br>

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
                <input type="text" id="start_time" name="start_time"><br>

                <label for="end_time">End time (seconds):</label>
                <input type="text" id="end_time" name="end_time"><br>

                <label for="upload_date">Upload date:</label>
                <input type="text" id="upload_data" name="upload_data" value="{{ existing_data['upload_date'] }}"><br>

                # <label for="fps">FPS:</label>
                # <select id="fps" name="fps">
                #     <option value="30.0">30.0</option>
                #     <option value="60.0">60.0</option>
                # </select><br>

                <label for="fps_list">FPS:</label>
                <input type="text" id="fps_list" name="fps_list" value="{{ existing_data['fps_list'] }}"><br>

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
                <input type="text" id="continent" name="continent" value="{{ existing_data['continent'] }}"><br>

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
            <h3>Embedded Video:</h3>
            <iframe id="embeddedVideo" width="100%" height="300" src="https://www.youtube.com/embed/{{ video_id }}" frameborder="0" allowfullscreen></iframe>
            <p>Current Time: <span id="currentTime">0</span> seconds</p>
            {% endif %}
        </div>
    </div>
</body>
</html>
"""

if __name__ == "__main__":
    app.run(debug=True)
