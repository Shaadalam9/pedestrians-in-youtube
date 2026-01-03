import common
from tqdm import tqdm
import pandas as pd
import numpy as np
from utils.core.metadata import MetaData

metadata_class = MetaData()


class Mapping_Enrich:
    def __init__(self) -> None:
        pass

    def add_speed_and_time_to_mapping(self, df_mapping, avg_speed_city, avg_time_city, avg_speed_country,
                                      avg_time_country, pedestrian_cross_city, pedestrian_cross_country,
                                      threshold=common.get_configs("min_crossing_detect")):
        """
        Adds city/country-level average speeds and/or times (day/night) to df_mapping DataFrame,
        depending on which dicts are provided. Missing columns are created and initialised with NaN.
        For country-level data, values are only added if pedestrian_cross_country[country_cond] < value.
        """
        configs = []
        if avg_speed_city is not None:
            configs.append(dict(
                label='city',
                avg_dict=avg_speed_city,
                value_type='speed',
                col_prefix='speed_crossing',
                key_parts=['city', 'lat', 'long', 'time_of_day'],
                get_state=True
            ))
        if avg_time_city is not None:
            configs.append(dict(
                label='city',
                avg_dict=avg_time_city,
                value_type='time',
                col_prefix='time_crossing',
                key_parts=['city', 'lat', 'long', 'time_of_day'],
                get_state=True
            ))
        if avg_speed_country is not None:
            configs.append(dict(
                label='country',
                avg_dict=avg_speed_country,
                value_type='speed',
                col_prefix='speed_crossing',
                key_parts=['country', 'time_of_day'],
                get_state=False
            ))
        if avg_time_country is not None:
            configs.append(dict(
                label='country',
                avg_dict=avg_time_country,
                value_type='time',
                col_prefix='time_crossing',
                key_parts=['country', 'time_of_day'],
                get_state=False
            ))

        for cfg in configs:
            label = cfg['label']
            avg_dict = cfg['avg_dict']
            col_prefix = cfg['col_prefix']
            get_state = cfg['get_state']  # noqa:F841

            # Prepare column names
            day_col = f"{col_prefix}_day_{label}"
            night_col = f"{col_prefix}_night_{label}"
            avg_col = f"{col_prefix}_day_night_{label}_avg"

            # Ensure columns exist and are initialised to np.nan
            for col in [day_col, night_col, avg_col]:
                if col not in df_mapping.columns:
                    df_mapping[col] = np.nan

            for key, value in tqdm(avg_dict.items(), desc=f"{label.capitalize()} {cfg['value_type'].capitalize()}s",
                                   total=len(avg_dict)):
                parts = key.split("_")
                if label == 'city':
                    city, lat, _, time_of_day = parts[0], parts[1], parts[2], int(parts[3])
                    state = metadata_class.get_value(df_mapping, "city", city, "lat", lat, "state")
                    mask = (
                        (df_mapping["city"] == city) &
                        ((df_mapping["state"] == state) | (pd.isna(df_mapping["state"]) & pd.isna(state)))
                    )
                else:  # country
                    # Parse country key
                    country, time_of_day = parts[0], int(parts[1])

                    # Check pedestrian_cross_country condition
                    cond_key = f"{country}_{time_of_day}"
                    cross_value = pedestrian_cross_country.get(cond_key, 0)
                    if cross_value <= threshold:
                        continue  # Skip if condition is not satisfied

                    mask = (df_mapping["country"] == country)
                if not time_of_day:
                    df_mapping.loc[mask, day_col] = float(value)
                else:
                    df_mapping.loc[mask, night_col] = float(value)

            # Calculate overall average column for each type
            df_mapping[avg_col] = np.where(
                (df_mapping[day_col] > 0) & (df_mapping[night_col] > 0),
                df_mapping[[day_col, night_col]].mean(axis=1),
                np.where(
                    df_mapping[day_col] > 0, df_mapping[day_col],
                    np.where(df_mapping[night_col] > 0, df_mapping[night_col], np.nan)
                )
            )

        return df_mapping
