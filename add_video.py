"""Adding new data to the mapping file."""
# by Pavlo Bazilinskyy <pavlo.bazilinskyy@gmail.com>
from flask import Flask, request, render_template
import pandas as pd
import os
import common
from pytubefix import YouTube
import webbrowser
from threading import Timer
import random
import requests
import ast
import re
import json
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable, GeocoderServiceError
from datetime import datetime

app = Flask(__name__)

FILE_PATH = common.get_configs("mapping")     # mapping file

# average height data from kaggle
height_data = pd.read_csv(os.path.join(common.root_dir, 'height_data.csv'))
# average age data from https://simplemaps.com/data/countries
age_data = pd.read_csv(os.path.join(common.root_dir, 'countries.csv'))


def format_city_label(city, state, country, iso3):
    if isinstance(state, str) and state.strip():
        return f"{city}, {state}, {country} ({iso3})"
    return f"{city}, {country} ({iso3})"


@app.route("/autocomplete/cities")
def autocomplete_cities():
    q = request.args.get("q", "").strip().lower()
    if len(q) < 2:
        return []

    df = load_csv(FILE_PATH)

    results = []
    seen = set()

    def normalize(s):
        return s.strip().lower()

    for _, row in df.iterrows():
        city = row.get("city")
        state = row.get("state")
        country = row.get("country")
        iso3 = row.get("iso3")

        if not all(isinstance(x, str) for x in [city, country]):
            continue

        city_norm = normalize(city)
        state_norm = normalize(state) if isinstance(state, str) else ""

        key = (city_norm, state_norm, country, iso3)

        if q in city_norm and key not in seen:
            seen.add(key)
            results.append({
                "city": city.strip(),
                "state": state if isinstance(state, str) else "",
                "country": country,
                "iso3": iso3,
                "label": format_city_label(
                    city.strip(),
                    state,
                    country,
                    iso3
                )
            })

        aka = row.get("city_aka")
        if isinstance(aka, str) and aka.startswith("[") and aka.endswith("]"):
            for item in aka[1:-1].split(","):
                name = item.strip()
                if not name:
                    continue

                name_norm = normalize(name)
                state_norm = normalize(state) if isinstance(state, str) else ""
                key = (name_norm, state_norm, country, iso3)

                if q in name_norm and key not in seen:
                    seen.add(key)
                    results.append({
                        "city": name,
                        "state": state if isinstance(state, str) else "",
                        "country": country,
                        "iso3": iso3,
                        "label": format_city_label(
                            name,
                            state,
                            country,
                            iso3
                        )
                    })

    results.sort(
        key=lambda x: (
            not normalize(x["city"]).startswith(q),
            normalize(x["city"])
        )
    )

    return results


def extract_city_autocomplete(file_path):
    """
    Extract unique city names from `city` and `city_aka` columns
    for autocomplete.
    """
    if not os.path.exists(file_path):
        return []

    df = pd.read_csv(file_path)

    cities = set()

    for _, row in df.iterrows():
        city = row.get("city")
        if isinstance(city, str) and city.strip():
            cities.add(city.strip())

        aka = row.get("city_aka")
        if isinstance(aka, str) and aka.strip():
            try:
                parsed = ast.literal_eval(aka)
                if isinstance(parsed, list):
                    for name in parsed:
                        if isinstance(name, str) and name.strip():
                            cities.add(name.strip())
            except Exception:
                pass

    return sorted(cities)


def load_csv(file_path):
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        return pd.DataFrame(columns=[
            'city', 'city_aka', 'state', 'country', 'iso3', 'videos',
            'time_of_day', 'vehicle_type', 'start_time', 'end_time', 'gmp',
            'population_city', 'population_country', 'traffic_mortality',
            'continent', 'literacy_rate', 'avg_height', 'med_age',
            'upload_date', 'channel', 'gini', 'traffic_index'
        ])


def save_csv(df, file_path):
    df.to_csv(file_path, index=False)


def compact_nested_list(value):
    """Serialise nested lists (ints) without spaces, eg [[0],[1,2]]."""
    try:
        return json.dumps(value, separators=(',', ':'))
    except Exception:
        return re.sub(r"\s+", "", str(value))


def compact_flat_list(value):
    """Serialise flat lists without spaces, eg [0,8,7]."""
    try:
        return json.dumps(value, separators=(',', ':'))
    except Exception:
        return re.sub(r"\s+", "", str(value))


def city_matches(row, city_input):
    city_input = city_input.strip().lower()
    main = str(row['city']).strip().lower()

    aka_raw = str(row.get('city_aka', '')).strip()
    aka_list = []
    if aka_raw.startswith("[") and aka_raw.endswith("]"):
        aka_list = [item.strip().lower() for item in aka_raw[1:-1].split(',') if item.strip()]

    return city_input == main or city_input in aka_list


def _is_missing(x):
    try:
        return x is None or (
            isinstance(x, float) and pd.isna(x)
        ) or (
            isinstance(x, str) and x.strip() in ["", "None", "nan"]
        )
    except Exception:
        return x is None


def _safe_literal_eval(val, default):
    """Safely parse stringified Python literals used in the mapping file."""
    if _is_missing(val):
        return default
    if isinstance(val, (list, dict, tuple)):
        return val
    s = str(val).strip()
    if not s:
        return default
    try:
        return ast.literal_eval(s)
    except Exception:
        return default


def _parse_videos_cell(videos_cell):
    """Return list of video ids from the mapping CSV videos cell."""
    if _is_missing(videos_cell):
        return []

    s = str(videos_cell).strip()
    if s.startswith('[') and s.endswith(']'):
        parsed = _safe_literal_eval(s, None)
        if isinstance(parsed, list):
            out = []
            for x in parsed:
                if _is_missing(x):
                    continue
                out.append(str(x).strip().strip('"').strip("'"))
            return [x for x in out if x]
        s = s[1:-1]

    parts = [p.strip().strip('"').strip("'") for p in s.split(',')]
    return [p for p in parts if p]


def _segments_for_video_in_row(row, video_id):
    """Return list of (start, end) segments for video_id within a single mapping row."""
    vids = _parse_videos_cell(row.get('videos', ''))
    if video_id not in vids:
        return []
    pos = vids.index(video_id)

    starts_all = _safe_literal_eval(row.get('start_time', ''), [])
    ends_all = _safe_literal_eval(row.get('end_time', ''), [])

    try:
        starts = starts_all[pos] if pos < len(starts_all) else []
        ends = ends_all[pos] if pos < len(ends_all) else []
    except Exception:
        starts, ends = [], []

    segs = []
    for st, et in zip(starts, ends):
        try:
            segs.append((int(st), int(et)))
        except Exception:
            continue
    return segs


def _segments_overlap_allow_touch(a_start, a_end, b_start, b_end):
    """Return True if two segments overlap."""
    a_start = int(a_start)
    a_end = int(a_end)
    b_start = int(b_start)
    b_end = int(b_end)
    return a_start < b_end and b_start < a_end


def _find_overlapping_segment(existing_segs, new_start, new_end):
    """Return the first (start, end) segment that overlaps the proposed segment, else None."""
    for st, et in existing_segs:
        try:
            if _segments_overlap_allow_touch(st, et, new_start, new_end):
                return (int(st), int(et))
        except Exception:
            continue
    return None


def find_overlap_across_mapping(video_hits, new_start, new_end, exclude_idxs=None):
    """Return info about the first overlapping segment found for this video anywhere in the mapping file."""
    if exclude_idxs:
        try:
            exclude = set(exclude_idxs)
        except Exception:
            exclude = set()
    else:
        exclude = set()

    for h in (video_hits or []):
        if not isinstance(h, dict):
            continue
        idx = h.get('idx')
        if idx in exclude:
            continue
        label = format_city_label(
            str(h.get('city', '')).strip(),
            h.get('state', ''),
            str(h.get('country', '')).strip(),
            str(h.get('iso3', '')).strip(),
        )
        for st, et in (h.get('segments', []) or []):
            try:
                if _segments_overlap_allow_touch(st, et, new_start, new_end):
                    return {'label': label, 'start': int(st), 'end': int(et), 'idx': idx}
            except Exception:
                continue
    return None


def find_video_occurrences(df, video_id):
    """Find every row in df that already contains video_id. Returns list of dicts."""
    hits = []
    if not video_id:
        return hits

    for idx, row in df.iterrows():
        vids = _parse_videos_cell(row.get('videos', ''))
        if video_id in vids:
            hits.append({
                'idx': idx,
                'city': row.get('city', ''),
                'state': row.get('state', ''),
                'country': row.get('country', ''),
                'iso3': row.get('iso3', ''),
                'segments': _segments_for_video_in_row(row, video_id),
            })
    return hits


def _format_segments(segs):
    if not segs:
        return "no segments saved"
    return ", ".join([f"{st} to {et}" for st, et in segs])


def build_video_occurrence_note(df, video_id, max_rows=8, exclude_idxs=None):
    """Build a user facing note listing where the video already exists and its segments."""
    hits = find_video_occurrences(df, video_id)
    if exclude_idxs:
        try:
            exclude = set(exclude_idxs)
        except Exception:
            exclude = set()
        hits = [h for h in hits if h.get('idx') not in exclude]
    if not hits:
        return ""

    parts = []
    for h in hits[:max_rows]:
        label = format_city_label(
            str(h.get('city', '')).strip(),
            h.get('state', ''),
            str(h.get('country', '')).strip(),
            str(h.get('iso3', '')).strip()
        )
        parts.append(f"{label}: {_format_segments(h.get('segments', []))}")

    extra = ""
    if len(hits) > max_rows:
        extra = f" (plus {len(hits) - max_rows} more)"

    return (
        "Video already exists elsewhere in the mapping file. Existing segments: "
        "<span style='color: red;'>" + "; ".join(parts) + extra + "</span>"
    )


def row_label(row):
    return format_city_label(
        str(row.get('city', '')).strip(),
        row.get('state', ''),
        str(row.get('country', '')).strip(),
        str(row.get('iso3', '')).strip()
    )


@app.route('/', methods=['GET', 'POST'])
def form():
    df = load_csv(FILE_PATH)
    message = ''
    city = ''
    city_aka = []
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
    med_age = ''
    upload_date_list = ''
    channel_list = ''
    vehicle_type_list = ''
    gini = ''
    traffic_index = ''
    upload_date_video = ''
    channel_video = ''
    yt_title = ''
    yt_upload_date = ''
    yt_channel = ''
    yt_description = ''
    start_time_video = []
    end_time_video = []
    vehicle_type_video = 0
    time_of_day_video = 0

    if request.method == 'POST':
        if 'fetch_data' in request.form:
            city = request.form.get('city')
            if city == 'None' or city == 'nan':
                city = None
            elif city is not None:
                city = city.strip()

            country = request.form.get('country')
            if country == 'None' or country == 'nan':
                country = None
            elif country is not None:
                country = country.strip()

            state = request.form.get('state')
            if state == 'None' or state == 'nan':
                state = None
            elif state is not None:
                state = state.strip()

            video_url = request.form.get('video_url')

            try:
                yt = YouTube(video_url)
                video_id = yt.video_id
                video_global_note = build_video_occurrence_note(df, video_id)

                yt_upload_date = yt.publish_date
                yt_channel = yt.channel_id

                for n in range(6):
                    try:
                        yt_description = yt.initial_data["engagementPanels"][n]["engagementPanelSectionListRenderer"]["content"]["structuredDescriptionContentRenderer"]["items"][1]["expandableVideoDescriptionBodyRenderer"]["attributedDescriptionBodyText"]["content"]  # noqa: E501
                    except Exception:
                        continue
            except Exception as e:
                return render_template(
                    "add_video.html",
                    message=f"Invalid YouTube URL: {e}",
                    df=df,
                    city=city,
                    country=country,
                    state=state,
                    video_url=video_url,
                    video_id=video_id,
                    existing_data=existing_data_row,
                    upload_date_video=upload_date_video,
                    channel_video=channel_video,
                    yt_title=yt_title,
                    yt_description=yt_description,
                    yt_upload_date=yt_upload_date
                )

            if state:
                filtered_df = df[(df['state'] == state) & (df['country'] == country)]
            else:
                filtered_df = df[df['country'] == country]

            existing_data = filtered_df[filtered_df.apply(lambda row: city_matches(row, city), axis=1)]

            if not existing_data.empty:
                message = "Entry for city found. You can update data."
                existing_data_row = existing_data.iloc[0].to_dict()
                city = existing_data_row.get('city')

                videos_list = existing_data_row.get('videos', '').split(',')
                videos_list = [video.strip('[]') for video in videos_list]

                upload_date_list = existing_data_row.get('upload_date', '').split(',')
                upload_date_list = [upload_date.strip('[]') for upload_date in upload_date_list]

                channel_list = existing_data_row.get('channel', '').split(',')
                channel_list = [channel.strip('[]') for channel in channel_list]

                vehicle_type_list = existing_data_row.get('vehicle_type', '').split(',')
                vehicle_type_list = [vehicle_type.strip('[]') for vehicle_type in vehicle_type_list]

                city_aka_list = existing_data_row.get('city_aka', '').split(',')
                city_aka_list = [city_aka.strip('[]') for city_aka in city_aka_list]

                if video_id in videos_list:
                    position = videos_list.index(video_id)
                    upload_date_video = upload_date_list[position].strip()
                    channel_video = channel_list[position].strip()

                    start_time_list = ast.literal_eval(existing_data_row.get('start_time', ''))
                    start_time_video = start_time_list[position]

                    end_time_list = ast.literal_eval(existing_data_row.get('end_time', ''))
                    end_time_video = end_time_list[position]

                    vehicle_type_video = vehicle_type_list[position]

                    time_of_day_list = ast.literal_eval(existing_data_row.get('time_of_day', ''))
                    time_of_day_video = time_of_day_list[position]
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

                lat, lon = get_coordinates(city, state, common.correct_country(country))
                existing_data_row = {
                    'city': city,
                    'city_aka': city_aka,
                    'country': country,
                    'iso3': iso3_code,
                    'lat': lat,
                    'lon': lon,
                    'state': state,
                    'videos': [],
                    'time_of_day': [],
                    'gmp': 0.0,
                    'population_city': int(get_city_population(city_data)),
                    'population_country': country_population,
                    'traffic_mortality': get_country_traffic_mortality(iso3_code),
                    'start_time': [],
                    'end_time': [],
                    'continent': get_country_continent(country_data),
                    'literacy_rate': get_country_literacy_rate(iso3_code),
                    'avg_height': get_country_average_height(iso3_code),
                    'med_age': get_country_median_age(iso2_code),
                    'upload_date': [],
                    'channel': [],
                    'vehicle_type': [],
                    'gini': get_country_gini(country_data),
                    'traffic_index': get_traffic_index_lat_lon(lat, lon)
                }

            if 'video_global_note' in locals() and video_global_note:
                message = (message + " " + video_global_note).strip()

        elif 'submit_data' in request.form:
            city = request.form.get('city')
            if city == 'None' or city == 'nan':
                city = None
            elif city is not None:
                city = city.strip()

            country = request.form.get('country')
            if country == 'None' or country == 'nan':
                country = None
            elif country is not None:
                country = country.strip()

            state = request.form.get('state')
            if state == 'None' or state == 'nan':
                state = None
            elif state is not None:
                state = state.strip()

            video_url = request.form.get('video_url')
            time_of_day = request.form.getlist('time_of_day')
            start_time = request.form.getlist('start_time')
            end_time = request.form.getlist('end_time')
            end_time_input = int(end_time[0])
            gmp = request.form.get('gmp')
            population_city = request.form.get('population_city')
            lat = request.form.get('lat')
            lon = request.form.get('lon')
            population_country = request.form.get('population_country')
            traffic_mortality = request.form.get('traffic_mortality')
            continent = request.form.get('continent')
            city_aka = request.form.get('city_aka')
            literacy_rate = request.form.get('literacy_rate')
            avg_height = request.form.get('avg_height')
            med_age = request.form.get('med_age')
            upload_date_video = request.form.get('upload_date_video')
            channel_video = request.form.get('channel_video')
            vehicle_type_video = request.form.get('vehicle_type')
            time_of_day_video = request.form.get('time_of_day')
            gini = request.form.get('gini')
            traffic_index = request.form.get('traffic_index')

            try:
                vehicle_type_video_int = int(vehicle_type_video)
            except (TypeError, ValueError):
                vehicle_type_video_int = None

            try:
                yt = YouTube(video_url)
                video_id = yt.video_id
                video_matches_anywhere = find_video_occurrences(df, video_id)
                yt_upload_date = yt.publish_date
                yt_channel = yt.channel_id

                for n in range(6):
                    try:
                        yt_description = yt.initial_data["engagementPanels"][n]["engagementPanelSectionListRenderer"]["content"]["structuredDescriptionContentRenderer"]["items"][1]["expandableVideoDescriptionBodyRenderer"]["attributedDescriptionBodyText"]["content"]  # noqa: E501
                    except Exception:
                        continue
            except Exception as e:
                return render_template(
                    "add_video.html",
                    message=f"Invalid YouTube URL: {e}",
                    df=df,
                    city=city,
                    country=country,
                    state=state,
                    video_url=video_url,
                    video_id=video_id,
                    existing_data=existing_data_row,
                    upload_date_video=upload_date_video,
                    channel_video=channel_video,
                    yt_title=yt_title,
                    yt_description=yt_description,
                    yt_upload_date=yt_upload_date,
                    yt_channel=yt_channel
                )

            if any(t not in ['0', '1'] for t in time_of_day):
                message = "Time of day must be either 0 or 1."
            elif vehicle_type_video_int not in range(13):
                message = "Type of vehicle must be one of: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12."
            elif any(int(et) <= int(st) for st, et in zip(start_time, end_time)):
                message = "End time must be larger than start time."
            else:
                if state:
                    check_existing = not df[
                        (df['city'] == city) &
                        (df['state'] == state) &
                        (df['country'] == country)
                    ].empty
                else:
                    check_existing = not df[
                        (df['city'] == city) &
                        (df['country'] == country)
                    ].empty

                current_idx = None
                if check_existing:
                    if state:
                        current_idx = df[
                            (df['city'] == city) &
                            (df['state'] == state) &
                            (df['country'] == country)
                        ].index[0]
                    else:
                        current_idx = df[
                            (df['city'] == city) &
                            (df['country'] == country)
                        ].index[0]

                global_note_for_display = ""
                duplicate_elsewhere = [h for h in video_matches_anywhere if h.get('idx') != current_idx]

                if duplicate_elsewhere:
                    global_note_for_display = build_video_occurrence_note(
                        df,
                        video_id,
                        exclude_idxs={current_idx} if current_idx is not None else None
                    )

                try:
                    new_start_global = int(start_time[-1])
                    new_end_global = int(end_time[-1])
                except Exception:
                    new_start_global = None
                    new_end_global = None

                overlap_global = None
                if new_start_global is not None and new_end_global is not None:
                    overlap_global = find_overlap_across_mapping(
                        video_matches_anywhere,
                        new_start_global,
                        new_end_global,
                        exclude_idxs={current_idx} if current_idx is not None else None
                    )

                if overlap_global:
                    next_start = int(overlap_global['end'])
                    message = (
                        f"Cannot add segment {new_start_global} to {new_end_global} because it overlaps existing segment "  # noqa: E501
                        f"{overlap_global['start']} to {overlap_global['end']} for this video in {overlap_global['label']}. "  # noqa: E501
                        f"Please use a start time of {next_start} or later."
                    )
                    if current_idx is not None:
                        existing_data_row = df.loc[current_idx].to_dict()

                if not message:
                    if check_existing:
                        if state:
                            idx = df[
                                (df['city'] == city) &
                                (df['state'] == state) &
                                (df['country'] == country)
                            ].index[0]
                        else:
                            idx = df[
                                (df['city'] == city) &
                                (df['country'] == country)
                            ].index[0]

                        videos_list = df.at[idx, 'videos'].split(',') if pd.notna(df.at[idx, 'videos']) else []
                        videos_list = [video.strip('[]') for video in videos_list]

                        time_of_day_list = eval(df.at[idx, 'time_of_day']) if pd.notna(df.at[idx, 'time_of_day']) else []  # noqa: E501
                        start_time_list = eval(df.at[idx, 'start_time']) if pd.notna(df.at[idx, 'start_time']) else []
                        end_time_list = eval(df.at[idx, 'end_time']) if pd.notna(df.at[idx, 'end_time']) else []

                        upload_date_list = df.at[idx, 'upload_date'].split(',') if pd.notna(df.at[idx, 'upload_date']) else []  # noqa: E501
                        upload_date_list = [upload_date.strip('[]') for upload_date in upload_date_list]

                        channel_list = df.at[idx, 'channel'].split(',') if pd.notna(df.at[idx, 'channel']) else []
                        channel_list = [channel.strip('[]') for channel in channel_list]

                        vehicle_type_list = df.at[idx, 'vehicle_type'].split(',') if pd.notna(df.at[idx, 'vehicle_type']) else []  # noqa: E501
                        vehicle_type_list = [vehicle_type.strip('[]') for vehicle_type in vehicle_type_list]

                        if video_id not in videos_list:
                            videos_list.append(video_id)
                            video_index = videos_list.index(video_id)
                            time_of_day_list.append([int(time_of_day[-1])])
                            start_time_list.append([int(start_time[-1])])
                            end_time_list.append([int(end_time[-1])])

                            if upload_date_video != 'None' and upload_date_video:
                                upload_date_list.append(int(upload_date_video))
                            else:
                                upload_date_list.append(None)

                            channel_list.append(channel_video)
                            vehicle_type_list.append(vehicle_type_video_int)
                        else:
                            video_index = videos_list.index(video_id)
                            new_start = int(start_time[-1])
                            new_end = int(end_time[-1])
                            existing_segs = list(zip(start_time_list[video_index], end_time_list[video_index]))
                            overlap = _find_overlapping_segment(existing_segs, new_start, new_end)

                            if overlap:
                                next_start = int(overlap[1])
                                message = (
                                    f"Cannot add segment {new_start} to {new_end} because it overlaps "
                                    f"existing segment {overlap[0]} to {overlap[1]} for this video in this city. "
                                    f"Please use a start time of {next_start} or later."
                                )
                                existing_data_row = df.loc[idx].to_dict()
                                start_time_video = start_time_list[video_index]
                                end_time_video = end_time_list[video_index]
                                time_of_day_video = time_of_day_list[video_index]

                                if not upload_date_video and yt_upload_date:
                                    upload_date_video = yt_upload_date.strftime('%d%m%Y')

                                if not channel_video and yt_channel:
                                    channel_video = yt_channel
                                elif not channel_video:
                                    channel_video = 'None'

                                time_of_day_last = extract_last_int(time_of_day_video)
                                time_of_day_last = int(time_of_day_last) if time_of_day_last is not None else None

                                return render_template(
                                    "add_video.html",
                                    message=message,
                                    df=df,
                                    city=city,
                                    country=country,
                                    state=state,
                                    video_url=video_url,
                                    video_id=video_id,
                                    existing_data=existing_data_row,
                                    upload_date_video=upload_date_video,
                                    channel_video=channel_video,
                                    timestamp=end_time_input,
                                    yt_title=yt_title,
                                    yt_description=yt_description,
                                    yt_upload_date=yt_upload_date,
                                    yt_channel=yt_channel,
                                    start_time_video=start_time_video,
                                    end_time_video=end_time_video,
                                    vehicle_type_video=vehicle_type_video_int,
                                    time_of_day_video=time_of_day_video,
                                    time_of_day_last=time_of_day_last
                                )

                            time_of_day_list[video_index].append(int(time_of_day[-1]))
                            start_time_list[video_index].append(new_start)
                            end_time_list[video_index].append(new_end)

                            if upload_date_video != 'None' and upload_date_video:
                                upload_date_list[video_index] = int(upload_date_video)
                            else:
                                upload_date_list[video_index] = None

                            channel_list[video_index] = channel_video
                            vehicle_type_list[video_index] = vehicle_type_video_int

                        start_time_video = start_time_list[video_index]
                        end_time_video = end_time_list[video_index]
                        time_of_day_video = time_of_day_list[video_index]

                        df.at[idx, 'videos'] = '[' + ','.join(videos_list) + ']'
                        df.at[idx, 'time_of_day'] = compact_nested_list(time_of_day_list)
                        df.at[idx, 'start_time'] = compact_nested_list(start_time_list)
                        df.at[idx, 'end_time'] = compact_nested_list(end_time_list)
                        df.at[idx, 'gmp'] = to_float_safe(gmp)
                        df.at[idx, 'population_city'] = to_int_safe(population_city)
                        df.at[idx, 'population_country'] = to_int_safe(population_country)
                        df.at[idx, 'traffic_mortality'] = to_float_safe(traffic_mortality)
                        df.at[idx, 'continent'] = continent
                        df.at[idx, 'city_aka'] = city_aka
                        df.at[idx, 'literacy_rate'] = to_float_safe(literacy_rate)
                        df.at[idx, 'avg_height'] = to_float_safe(avg_height)
                        df.at[idx, 'med_age'] = to_float_safe(med_age)
                        df.at[idx, 'lat'] = to_float_safe(lat)
                        df.at[idx, 'lon'] = to_float_safe(lon)

                        for i in range(len(upload_date_list)):
                            if upload_date_list[i] != 'None' and upload_date_list[i]:
                                upload_date_list[i] = int(upload_date_list[i])
                            else:
                                upload_date_list[i] = None
                        df.at[idx, 'upload_date'] = compact_flat_list(upload_date_list)

                        for i in range(len(channel_list)):
                            if channel_list[i] != 'None':
                                channel_list[i] = channel_list[i]
                        df.at[idx, 'channel'] = compact_flat_list(channel_list)

                        vehicle_type_list = [int(x) for x in vehicle_type_list]
                        df.at[idx, 'vehicle_type'] = compact_flat_list(vehicle_type_list)

                        if gini:
                            df.at[idx, 'gini'] = float(gini)
                        else:
                            df.at[idx, 'gini'] = 0.0

                        if traffic_index:
                            df.at[idx, 'traffic_index'] = float(traffic_index)
                        else:
                            df.at[idx, 'traffic_index'] = 0.0

                    else:
                        new_row = {
                            'id': int(df.iloc[-1]['id'] + 1),
                            'city': city,
                            'city_aka': city_aka,
                            'state': state,
                            'country': country,
                            'iso3': common.get_iso3_country_code(common.correct_country(country)),
                            'lat': lat,
                            'lon': lon,
                            'videos': '[' + video_id + ']',
                            'time_of_day': compact_nested_list([[int(x) for x in time_of_day]]),
                            'start_time': compact_nested_list([[int(x) for x in start_time]]),
                            'end_time': compact_nested_list([[int(x) for x in end_time]]),
                            'gmp': gmp,
                            'population_city': population_city,
                            'population_country': population_country,
                            'traffic_mortality': traffic_mortality,
                            'continent': continent,
                            'literacy_rate': literacy_rate,
                            'avg_height': avg_height,
                            'med_age': med_age,
                            'upload_date': compact_flat_list([int(upload_date_video)]) if upload_date_video and upload_date_video != 'None' else '[null]',  # noqa: E501
                            'channel': compact_flat_list([channel_video.strip()]) if channel_video else '[None]',
                            'vehicle_type': compact_flat_list([vehicle_type_video_int]),
                            'gini': gini,
                            'traffic_index': traffic_index,
                        }
                        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                        start_time_video = [int(x) for x in start_time]
                        end_time_video = [int(x) for x in end_time]
                        time_of_day_video = [int(x) for x in time_of_day]

                    save_csv(df, FILE_PATH)
                    message = "Video added or updated successfully."
                    if global_note_for_display:
                        message = message + " " + global_note_for_display

    if state:
        if city and state and country:
            existing_data = df[(df['city'] == city) & (df['state'] == state) & (df['country'] == country)]
            if not existing_data.empty:
                existing_data_row = existing_data.iloc[0].to_dict()
    else:
        if city and country:
            existing_data = df[(df['city'] == city) & (df['country'] == country)]
            if not existing_data.empty:
                existing_data_row = existing_data.iloc[0].to_dict()

    if not upload_date_video and yt_upload_date:
        upload_date_video = yt_upload_date.strftime('%d%m%Y')

    if not channel_video and yt_channel:
        channel_video = yt_channel
    elif not channel_video:
        channel_video = 'None'

    vehicle_type_video = int(vehicle_type_video) if vehicle_type_video is not None else None
    time_of_day_last = extract_last_int(time_of_day_video)
    time_of_day_last = int(time_of_day_last) if time_of_day_last is not None else None

    return render_template(
        "add_video.html",
        message=message,
        df=df,
        city=city,
        country=country,
        state=state,
        video_url=video_url,
        video_id=video_id,
        existing_data=existing_data_row,
        upload_date_video=upload_date_video,
        channel_video=channel_video,
        timestamp=end_time_input,
        yt_title=yt_title,
        yt_description=yt_description,
        yt_upload_date=yt_upload_date,
        yt_channel=yt_channel,
        start_time_video=start_time_video,
        end_time_video=end_time_video,
        vehicle_type_video=vehicle_type_video,
        time_of_day_video=time_of_day_video,
        time_of_day_last=time_of_day_last
    )


@app.route('/fetch_video', methods=['POST'])
def fetch_video():
    return form()


@app.route('/submit_data', methods=['POST'])
def submit_data():
    return form()


def get_country_data(iso3_code):
    try:
        api_url = f"https://restcountries.com/v3.1/alpha/{iso3_code}"
        response = requests.get(api_url)
        if response.status_code == 200:
            country_data = response.json()
            return country_data
        else:
            return None
    except Exception as e:
        print(f"Error fetching country data: {e}")
        return None


def get_country_continent(country_data):
    if country_data:
        return country_data[0]['continents'][0]
    else:
        return ''


def get_country_population(country_data):
    if country_data:
        return country_data[0]['population']
    else:
        return 0.0


def get_country_gini(country_data):
    if country_data:
        gini_data = country_data[0].get('gini', {})
        if gini_data:
            return list(gini_data.values())[0]
        else:
            return 0.0
    else:
        return 0.0


def get_country_literacy_rate(iso3_code):
    try:
        api_url = f"http://api.worldbank.org/v2/country/{iso3_code}/indicator/SE.ADT.LITR.ZS?format=json"
        response = requests.get(api_url)
        if response.status_code == 200:
            data = response.json()
            if data and len(data) > 1 and data[1]:
                for entry in data[1]:
                    if entry['value'] is not None:
                        return round(entry['value'], 2)
            return 0.0
        else:
            return 0.0
    except Exception as e:
        print(f"Error fetching literacy rate: {e}")
        return 0.0


def get_country_traffic_mortality(iso3_code):
    try:
        api_url = f"http://api.worldbank.org/v2/country/{iso3_code}/indicator/SH.STA.TRAF.P5?format=json"
        response = requests.get(api_url)
        if response.status_code == 200:
            data = response.json()
            if data and len(data) > 1 and data[1]:
                for entry in data[1]:
                    if entry['value'] is not None:
                        return round(entry['value'], 2)
            return 0.0
    except Exception as e:
        print(f"Error fetching traffic mortality rate: {e}")
        return 0.0


def get_city_data(city, country_code):
    """
    Get economic or city related data from the Geonames API
    """
    url = f"http://api.geonames.org/searchJSON?q={city}&country={country_code}&username={common.get_secrets('geonames_username')}"  # noqa: E501
    try:
        response = requests.get(url)
    except requests.exceptions.ConnectionError as e:
        print(f"Connection error while getting city data for {city}, {country_code}: {e}.")
        return None

    if response.status_code == 200:
        return response.json()
    else:
        return None


def get_city_population(city_data):
    """
    Get population from Geonames city data
    """
    if not city_data:
        return 0

    if not isinstance(city_data, dict):
        return 0

    geonames = city_data.get("geonames")

    if isinstance(geonames, list) and len(geonames) > 0:
        return geonames[0].get("population", 0)

    return 0


def get_country_average_height(iso3_code):
    try:
        row = height_data[height_data['cca3'].str.lower() == iso3_code.lower()]
        if not row.empty:
            male_height = row.iloc[0]['meanHeightMale']
            female_height = row.iloc[0]['meanHeightFemale']
            avg_height = (male_height + female_height) / 2
            return round(avg_height)
        else:
            return 0.0
    except Exception as e:
        print(f"Error fetching height data: {e}")
        return 0.0


def get_country_median_age(iso2_code):
    try:
        row = age_data[age_data['iso2'].str.lower() == iso2_code.lower()]
        if not row.empty:
            return row.iloc[0]['median_age']
        else:
            return 0.0
    except Exception as e:
        print(f"Error fetching height data: {e}")
        return 0.0


def get_gmp(city: str, state: str, iso3: str) -> float:
    """
    Fetches Gross Metropolitan Product (GMP) for a given city, state, and ISO3 country code.
    """
    if iso3.upper() == "USA":
        url = "https://apps.bea.gov/api/data/"
        params = {
            "UserID": common.get_secrets('bea_api_key'),
            "method": "GetData",
            "datasetname": "Regional",
            "TableName": "CAGDP2",
            "LineCode": "1",
            "GeoFIPS": "MSA",
            "Year": "2022",
            "ResultFormat": "json"
        }
        response = requests.get(url, params=params)
        data = response.json()

        for entry in data.get("BEAAPI", {}).get("Results", {}).get("Data", []):
            if city.lower() in entry["GeoName"].lower():
                return float(entry["DataValue"])

    else:
        url = f"https://stats.oecd.org/SDMX-JSON/data/CITIES/GDP.METRO.{iso3.upper()}?json-lang=en"
        response = requests.get(url)
        data = response.json()

        for key, value in data.get("dataSets", [{}])[0].get("observations", {}).items():
            if city.lower() in key.lower():
                return float(value[0])

    return None


def get_traffic_index_lat_lon(lat, lon, api="tomtom"):
    if api == "tomtom":
        url = f"https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json?key={common.get_secrets('tomtom_api_key')}&point={lat},{lon}"  # noqa: E501
        try:
            response = requests.get(url)

            if response.status_code == 200:
                data = response.json()
                if "flowSegmentData" in data:
                    current_speed = data["flowSegmentData"]["currentSpeed"]
                    free_flow_speed = data["flowSegmentData"]["freeFlowSpeed"]
                    try:
                        traffic_index = round((1 - current_speed / free_flow_speed) * 100, 2)
                    except ZeroDivisionError:
                        traffic_index = 0.0
                    return traffic_index
                else:
                    return 0.0
            else:
                print(f"Error fetching traffic index for {lat}, {lon}: {response.status_code}")
                return 0.0
        except ConnectionError as e:
            print(f"An error occurred: {e}")
            return 0.0
    elif api == "trafiklab":
        url = f"https://api.trafiklab.se/v1/trafficindex?lat={lat}&lon={lon}&apikey={common.get_secrets('trafiklab_api_key')}"  # noqa: E501

        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()

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
    if state:
        url = f"https://www.numbeo.com/api/traffic?api_key={common.get_secrets('numbeo_api_key')}&city={city}&state={state}&country={country}"  # noqa: E501
    else:
        url = f"https://www.numbeo.com/api/traffic?api_key={common.get_secrets('numbeo_api_key')}&city={city}&country={country}"  # noqa: E501
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

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
    """Get city coordinates."""
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    user_agent = f"my_geocoding_script_{current_time}"
    geolocator = Nominatim(user_agent=user_agent)

    try:
        if state and str(state).lower() != 'nan':
            location_query = f"{city}, {state}, {country}"
        else:
            location_query = f"{city}, {country}"

        location = geolocator.geocode(location_query, timeout=2)

        if location:
            return location.latitude, location.longitude
        else:
            print(f"Failed to geocode {location_query}")
            return None, None

    except GeocoderTimedOut:
        print(f"Geocoding timed out for {location_query}.")
        return None, None
    except GeocoderUnavailable:
        print(f"Geocoding server could not be reached for {location_query}.")
        return None, None
    except GeocoderServiceError:
        print(f"Non successful status for {location_query}.")
        return None, None


def extract_last_int(value):
    if isinstance(value, int):
        return value

    if isinstance(value, list):
        if len(value) > 0:
            try:
                return int(value[-1])
            except Exception:
                return None
        return None

    if isinstance(value, str):
        val = value.strip()

        try:
            return int(val)
        except Exception:
            pass

        try:
            parsed = ast.literal_eval(val)
            if isinstance(parsed, list) and len(parsed) > 0:
                return int(parsed[-1])
            return None
        except Exception:
            return None

    return None


def to_int_safe(value):
    if value is None:
        return 0
    try:
        f = float(value)
        return int(f)
    except Exception:
        return 0


def to_float_safe(value):
    if value is None:
        return 0.0
    try:
        return float(value)
    except Exception:
        return 0.0


if __name__ == "__main__":
    port = 5000 + random.randint(0, 999)
    url = "http://127.0.0.1:{0}".format(port)
    Timer(1.25, lambda: webbrowser.open(url)).start()
    app.run(port=port, debug=False)
