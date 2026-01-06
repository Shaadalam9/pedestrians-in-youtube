"""
Project module setup.

Authors:
- Shadab Alam <md_shadab_alam@outlook.com>
- Pavlo Bazilinskyy <pavlo.bazilinskyy@gmail.com>
"""

from __future__ import annotations

import ast
import math
import os
import pickle
import re
import warnings
from typing import Set, Optional, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

import common
from custom_logger import CustomLogger
from logmod import logs
from utils.analytics.aggregation import Aggregation
from utils.analytics.durations import Duration
from utils.analytics.events import Events
from utils.analytics.geo import Geo
from utils.analytics.io import IO
from utils.analytics.mapping_enrichment import Mapping_Enrich
from utils.analytics.metrics_cache import Metrics_cache
from utils.core.dataset_stats import Dataset_Stats
from utils.core.tools import Tools
from utils.crossing.detection import Detection
from utils.crossing.metrics import Metrics
from utils.plotting.bivariate import Bivariate
from utils.plotting.correlations import Correlations
from utils.plotting.crossings import Crossings
from utils.plotting.distributions import Distributions
from utils.plotting.maps import Maps
from utils.plotting.stacked import Stacked

# ---------------------------------------------------------------------
# Global configuration
# ---------------------------------------------------------------------

# Suppress a specific FutureWarning emitted by plotly.
warnings.filterwarnings("ignore", category=FutureWarning, module="plotly")

logs(show_level=common.get_configs("logger_level"), show_color=True)
logger = CustomLogger(__name__)  # use custom logger

# ---------------------------------------------------------------------
# Class instances (singletons for this module)
# ---------------------------------------------------------------------

tools = Tools()
maps = Maps()
bivariate = Bivariate()
stacked = Stacked()
distribution = Distributions()

dataset_stats = Dataset_Stats()
metrics_cache = Metrics_cache()
analytics_IO = IO()
duration = Duration()
mapping_enrich = Mapping_Enrich()
detection = Detection()
metrics = Metrics()
events = Events()
crossing = Crossings()
geo = Geo()
correlation = Correlations()
aggregation = Aggregation()

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

# File to store the city coordinates.
file_results: str = "results.pickle"

video_paths = common.get_configs("videos")

# Common junk files/folders to ignore.
MISC_FILES: Set[str] = {"DS_Store", "seg", "bbox"}


class Analysis():

    def __init__(self) -> None:
        pass

    # Emoji flag mapping for ISO3 codes
    iso3_to_flag = {
        'ABW': '🇦🇼',  # Aruba
        'AFG': '🇦🇫',  # Afghanistan
        'AGO': '🇦🇴',  # Angola
        'AIA': '🇦🇮',  # Anguilla
        'ALA': '🇦🇽',  # Åland Islands
        'ALB': '🇦🇱',  # Albania
        'AND': '🇦🇩',  # Andorra
        'ARE': '🇦🇪',  # United Arab Emirates
        'ARG': '🇦🇷',  # Argentina
        'ARM': '🇦🇲',  # Armenia
        'ASM': '🇦🇸',  # American Samoa
        'ATA': '🇦🇶',  # Antarctica
        'ATF': '🏳️',  # French Southern Territories (no Unicode flag)
        'ATG': '🇦🇬',  # Antigua and Barbuda
        'AUS': '🇦🇺',  # Australia
        'AUT': '🇦🇹',  # Austria
        'AZE': '🇦🇿',  # Azerbaijan
        'BDI': '🇧🇮',  # Burundi
        'BEL': '🇧🇪',  # Belgium
        'BEN': '🇧🇯',  # Benin
        'BES': '🇧🇶',  # Bonaire, Sint Eustatius and Saba
        'BFA': '🇧🇫',  # Burkina Faso
        'BGD': '🇧🇩',  # Bangladesh
        'BGR': '🇧🇬',  # Bulgaria
        'BHR': '🇧🇭',  # Bahrain
        'BHS': '🇧🇸',  # Bahamas
        'BIH': '🇧🇦',  # Bosnia and Herzegovina
        'BLM': '🇧🇱',  # Saint Barthélemy
        'BLR': '🇧🇾',  # Belarus
        'BLZ': '🇧🇿',  # Belize
        'BMU': '🇧🇲',  # Bermuda
        'BOL': '🇧🇴',  # Bolivia
        'BRA': '🇧🇷',  # Brazil
        'BRB': '🇧🇧',  # Barbados
        'BRN': '🇧🇳',  # Brunei
        'BTN': '🇧🇹',  # Bhutan
        'BVT': '🏳️',  # Bouvet Island (no Unicode flag)
        'BWA': '🇧🇼',  # Botswana
        'CAF': '🇨🇫',  # Central African Republic
        'CAN': '🇨🇦',  # Canada
        'CCK': '🇨🇨',  # Cocos (Keeling) Islands
        'CHE': '🇨🇭',  # Switzerland
        'CHL': '🇨🇱',  # Chile
        'CHN': '🇨🇳',  # China
        'CIV': '🇨🇮',  # Côte d'Ivoire
        'CMR': '🇨🇲',  # Cameroon
        'COD': '🇨🇩',  # DR Congo
        'COG': '🇨🇬',  # Congo
        'COK': '🇨🇰',  # Cook Islands
        'COL': '🇨🇴',  # Colombia
        'COM': '🇰🇲',  # Comoros
        'CPV': '🇨🇻',  # Cape Verde
        'CRI': '🇨🇷',  # Costa Rica
        'CUB': '🇨🇺',  # Cuba
        'CUW': '🇨🇼',  # Curaçao
        'CXR': '🇨🇽',  # Christmas Island
        'CYM': '🇰🇾',  # Cayman Islands
        'CYP': '🇨🇾',  # Cyprus
        'CZE': '🇨🇿',  # Czechia
        'DEU': '🇩🇪',  # Germany
        'DJI': '🇩🇯',  # Djibouti
        'DMA': '🇩🇲',  # Dominica
        'DNK': '🇩🇰',  # Denmark
        'DOM': '🇩🇴',  # Dominican Republic
        'DZA': '🇩🇿',  # Algeria
        'ECU': '🇪🇨',  # Ecuador
        'EGY': '🇪🇬',  # Egypt
        'ERI': '🇪🇷',  # Eritrea
        'ESH': '🇪🇭',  # Western Sahara
        'ESP': '🇪🇸',  # Spain
        'EST': '🇪🇪',  # Estonia
        'ETH': '🇪🇹',  # Ethiopia
        'FIN': '🇫🇮',  # Finland
        'FJI': '🇫🇯',  # Fiji
        'FLK': '🇫🇰',  # Falkland Islands
        'FRA': '🇫🇷',  # France
        'FRO': '🇫🇴',  # Faroe Islands
        'FSM': '🇫🇲',  # Micronesia
        'GAB': '🇬🇦',  # Gabon
        'GBR': '🇬🇧',  # United Kingdom
        'GEO': '🇬🇪',  # Georgia
        'GGY': '🇬🇬',  # Guernsey
        'GHA': '🇬🇭',  # Ghana
        'GIB': '🇬🇮',  # Gibraltar
        'GIN': '🇬🇳',  # Guinea
        'GLP': '🇬🇵',  # Guadeloupe
        'GMB': '🇬🇲',  # Gambia
        'GNB': '🇬🇼',  # Guinea-Bissau
        'GNQ': '🇬🇶',  # Equatorial Guinea
        'GRC': '🇬🇷',  # Greece
        'GRD': '🇬🇩',  # Grenada
        'GRL': '🇬🇱',  # Greenland
        'GTM': '🇬🇹',  # Guatemala
        'GUF': '🇬🇫',  # French Guiana
        'GUM': '🇬🇺',  # Guam
        'GUY': '🇬🇾',  # Guyana
        'HKG': '🇭🇰',  # Hong Kong
        'HMD': '🇭🇲',  # Heard Island and McDonald Islands
        'HND': '🇭🇳',  # Honduras
        'HRV': '🇭🇷',  # Croatia
        'HTI': '🇭🇹',  # Haiti
        'HUN': '🇭🇺',  # Hungary
        'IDN': '🇮🇩',  # Indonesia
        'IMN': '🇮🇲',  # Isle of Man
        'IND': '🇮🇳',  # India
        'IOT': '🇮🇴',  # British Indian Ocean Territory
        'IRL': '🇮🇪',  # Ireland
        'IRN': '🇮🇷',  # Iran
        'IRQ': '🇮🇶',  # Iraq
        'ISL': '🇮🇸',  # Iceland
        'ISR': '🇮🇱',  # Israel
        'ITA': '🇮🇹',  # Italy
        'JAM': '🇯🇲',  # Jamaica
        'JEY': '🇯🇪',  # Jersey
        'JOR': '🇯🇴',  # Jordan
        'JPN': '🇯🇵',  # Japan
        'KAZ': '🇰🇿',  # Kazakhstan
        'KEN': '🇰🇪',  # Kenya
        'KGZ': '🇰🇬',  # Kyrgyzstan
        'KHM': '🇰🇭',  # Cambodia
        'KIR': '🇰🇮',  # Kiribati
        'KNA': '🇰🇳',  # Saint Kitts and Nevis
        'KOR': '🇰🇷',  # South Korea
        'KWT': '🇰🇼',  # Kuwait
        'LAO': '🇱🇦',  # Laos
        'LBN': '🇱🇧',  # Lebanon
        'LBR': '🇱🇷',  # Liberia
        'LBY': '🇱🇾',  # Libya
        'LCA': '🇱🇨',  # Saint Lucia
        'LIE': '🇱🇮',  # Liechtenstein
        'LKA': '🇱🇰',  # Sri Lanka
        'LSO': '🇱🇸',  # Lesotho
        'LTU': '🇱🇹',  # Lithuania
        'LUX': '🇱🇺',  # Luxembourg
        'LVA': '🇱🇻',  # Latvia
        'MAC': '🇲🇴',  # Macao
        'MAF': '🇲🇫',  # Saint Martin
        'MAR': '🇲🇦',  # Morocco
        'MCO': '🇲🇨',  # Monaco
        'MDA': '🇲🇩',  # Moldova
        'MDG': '🇲🇬',  # Madagascar
        'MDV': '🇲🇻',  # Maldives
        'MEX': '🇲🇽',  # Mexico
        'MHL': '🇲🇭',  # Marshall Islands
        'MKD': '🇲🇰',  # North Macedonia
        'MLI': '🇲🇱',  # Mali
        'MLT': '🇲🇹',  # Malta
        'MMR': '🇲🇲',  # Myanmar
        'MNE': '🇲🇪',  # Montenegro
        'MNG': '🇲🇳',  # Mongolia
        'MNP': '🇲🇵',  # Northern Mariana Islands
        'MOZ': '🇲🇿',  # Mozambique
        'MRT': '🇲🇷',  # Mauritania
        'MSR': '🇲🇸',  # Montserrat
        'MTQ': '🇲🇶',  # Martinique
        'MUS': '🇲🇺',  # Mauritius
        'MWI': '🇲🇼',  # Malawi
        'MYS': '🇲🇾',  # Malaysia
        'MYT': '🇾🇹',  # Mayotte
        'NAM': '🇳🇦',  # Namibia
        'NCL': '🇳🇨',  # New Caledonia
        'NER': '🇳🇪',  # Niger
        'NFK': '🇳🇫',  # Norfolk Island
        'NGA': '🇳🇬',  # Nigeria
        'NIC': '🇳🇮',  # Nicaragua
        'NIU': '🇳🇺',  # Niue
        'NLD': '🇳🇱',  # Netherlands
        'NOR': '🇳🇴',  # Norway
        'NPL': '🇳🇵',  # Nepal
        'NRU': '🇳🇷',  # Nauru
        'NZL': '🇳🇿',  # New Zealand
        'OMN': '🇴🇲',  # Oman
        'PAK': '🇵🇰',  # Pakistan
        'PAN': '🇵🇦',  # Panama
        'PCN': '🇵🇳',  # Pitcairn Islands
        'PER': '🇵🇪',  # Peru
        'PHL': '🇵🇭',  # Philippines
        'PLW': '🇵🇼',  # Palau
        'PNG': '🇵🇬',  # Papua New Guinea
        'POL': '🇵🇱',  # Poland
        'PRI': '🇵🇷',  # Puerto Rico
        'PRK': '🇰🇵',  # North Korea
        'PRT': '🇵🇹',  # Portugal
        'PRY': '🇵🇾',  # Paraguay
        'PSE': '🇵🇸',  # Palestine
        'PYF': '🇵🇫',  # French Polynesia
        'QAT': '🇶🇦',  # Qatar
        'REU': '🇷🇪',  # Réunion
        'ROU': '🇷🇴',  # Romania
        'RUS': '🇷🇺',  # Russia
        'RWA': '🇷🇼',  # Rwanda
        'SAU': '🇸🇦',  # Saudi Arabia
        'SDN': '🇸🇩',  # Sudan
        'SEN': '🇸🇳',  # Senegal
        'SGP': '🇸🇬',  # Singapore
        'SGS': '🏳️',  # South Georgia & South Sandwich Islands (no Unicode flag)
        'SHN': '🇸🇭',  # Saint Helena
        'SJM': '🏳️',  # Svalbard and Jan Mayen (no Unicode flag)
        'SLB': '🇸🇧',  # Solomon Islands
        'SLE': '🇸🇱',  # Sierra Leone
        'SLV': '🇸🇻',  # El Salvador
        'SMR': '🇸🇲',  # San Marino
        'SOM': '🇸🇴',  # Somalia
        'SPM': '🇵🇲',  # Saint Pierre and Miquelon
        'SRB': '🇷🇸',  # Serbia
        'SSD': '🇸🇸',  # South Sudan
        'STP': '🇸🇹',  # São Tomé and Príncipe
        'SUR': '🇸🇷',  # Suriname
        'SVK': '🇸🇰',  # Slovakia
        'SVN': '🇸🇮',  # Slovenia
        'SWE': '🇸🇪',  # Sweden
        'SWZ': '🇸🇿',  # Eswatini
        'SXM': '🇸🇽',  # Sint Maarten
        'SYC': '🇸🇨',  # Seychelles
        'SYR': '🇸🇾',  # Syria
        'TCA': '🇹🇨',  # Turks and Caicos Islands
        'TCD': '🇹🇩',  # Chad
        'TGO': '🇹🇬',  # Togo
        'THA': '🇹🇭',  # Thailand
        'TJK': '🇹🇯',  # Tajikistan
        'TKL': '🇹🇰',  # Tokelau
        'TKM': '🇹🇲',  # Turkmenistan
        'TLS': '🇹🇱',  # Timor-Leste
        'TON': '🇹🇴',  # Tonga
        'TTO': '🇹🇹',  # Trinidad and Tobago
        'TUN': '🇹🇳',  # Tunisia
        'TUR': '🇹🇷',  # Turkey
        'TUV': '🇹🇻',  # Tuvalu
        'TWN': '🇹🇼',  # Taiwan
        'TZA': '🇹🇿',  # Tanzania
        'UGA': '🇺🇬',  # Uganda
        'UKR': '🇺🇦',  # Ukraine
        'UMI': '🇺🇲',  # U.S. Minor Outlying Islands
        'URY': '🇺🇾',  # Uruguay
        'USA': '🇺🇸',  # United States
        'UZB': '🇺🇿',  # Uzbekistan
        'VAT': '🇻🇦',  # Vatican City
        'VCT': '🇻🇨',  # Saint Vincent and the Grenadines
        'VEN': '🇻🇪',  # Venezuela
        'VGB': '🇻🇬',  # British Virgin Islands
        'VIR': '🇻🇮',  # U.S. Virgin Islands
        'VNM': '🇻🇳',  # Vietnam
        'VUT': '🇻🇺',  # Vanuatu
        'WLF': '🇼🇫',  # Wallis and Futuna
        'WSM': '🇼🇸',  # Samoa
        'XKX': '🇽🇰',  # Kosovo
        'YEM': '🇾🇪',  # Yemen
        'ZAF': '🇿🇦',  # South Africa
        'ZMB': '🇿🇲',  # Zambia
        'ZWE': '🇿🇼',  # Zimbabwe
    }

    # vehicle type mapping
    vehicle_map = {
        0: "Car",
        1: "Bus",
        2: "Truck",
        3: "Two-wheeler",
        4: "Bicycle",
        5: "Automated car",
        6: "Automated bus",
        7: "Automated truck",
        8: "Automated two-wheeler",
        9: "Electric scooter",
        10: "Non-electric scooter",
        11: "Monowheel/unicycle",
        12: "Pedestrian"
    }

    # time of day mapping
    time_map = {0: "Day", 1: "Night"}


analysis_class = Analysis()

# Execute analysis
if __name__ == "__main__":
    logger.info("Analysis started.")

    if os.path.exists(file_results) and not common.get_configs('always_analyse'):
        # Load the data from the pickle file
        with open(file_results, 'rb') as file:
            (data,                                          # 0
             person_counter,                                # 1
             bicycle_counter,                               # 2
             car_counter,                                   # 3
             motorcycle_counter,                            # 4
             bus_counter,                                   # 5
             truck_counter,                                 # 6
             cellphone_counter,                             # 7
             traffic_light_counter,                         # 8
             stop_sign_counter,                             # 9
             pedestrian_cross_city,                         # 10
             pedestrian_crossing_count,                     # 11
             person_city,                                   # 12
             bicycle_city,                                  # 13
             car_city,                                      # 14
             motorcycle_city,                               # 15
             bus_city,                                      # 16
             truck_city,                                    # 17
             cross_evnt_city,                               # 18
             vehicle_city,                                  # 19
             cellphone_city,                                # 20
             traffic_sign_city,                             # 21
             all_speed,                                     # 22
             all_time,                                      # 23
             avg_time_city,                                 # 24
             avg_speed_city,                                # 25
             df_mapping,                                    # 26
             avg_speed_country,                             # 27
             avg_time_country,                              # 28
             crossings_with_traffic_equipment_city,         # 29
             crossings_without_traffic_equipment_city,      # 30
             crossings_with_traffic_equipment_country,      # 31
             crossings_without_traffic_equipment_country,   # 32
             min_max_speed,                                 # 33
             min_max_time,                                  # 34
             pedestrian_cross_country,                      # 35
             all_speed_city,                                # 36
             all_time_city,                                 # 37
             all_speed_country,                             # 38
             all_time_country,                              # 39
             df_mapping_raw,                                # 40
             pedestrian_cross_city_all,                     # 41
             pedestrian_cross_country_all                   # 42
             ) = pickle.load(file)

        logger.info("Loaded analysis results from pickle file.")
    else:
        # Store the mapping file
        df_mapping = pd.read_csv(common.get_configs("mapping"))

        # Produce map with all data
        df = df_mapping.copy()  # copy df to manipulate for output
        df['state'] = df['state'].fillna('NA')  # Set state to NA

        # Sort by continent and city, both in ascending order
        df = df.sort_values(by=["continent", "city"], ascending=[True, True])

        # Count of videos
        df['video_count'] = df['videos'].apply(lambda x: len(x.strip('[]').split(',')) if pd.notna(x) else 0)

        # Total amount of seconds in segments
        def flatten(lst):
            """Flattens nested lists like [[1, 2], [3, 4]] -> [1, 2, 3, 4]"""
            return [item for sublist in lst for item in (sublist if isinstance(sublist, list) else [sublist])]

        def compute_total_time(row):
            try:
                start_times = flatten(ast.literal_eval(row['start_time']))
                end_times = flatten(ast.literal_eval(row['end_time']))
                return sum(e - s for s, e in zip(start_times, end_times))
            except Exception as e:
                logger.error(f"Error in row {row['id']}: {e}")
                return 0

        df['total_time'] = df.apply(compute_total_time, axis=1)

        # create flag_city column
        df['flag_city'] = df.apply(lambda row: f"{analysis_class.iso3_to_flag.get(row['iso3'], '🏳️')} {row['city']}",
                                   axis=1)
        # Create a new country label with emoji flag + country name
        df["flag_country"] = df.apply(
            lambda row: f"{analysis_class.iso3_to_flag.get(row['iso3'], '🏳️')} {row['country']}",
            axis=1
        )

        # Data to avoid showing on hover in scatter plots
        columns_remove = ['videos', 'time_of_day', 'start_time', 'end_time', 'upload_date', 'vehicle_type', 'channel',
                          'display_label', 'flag_city', 'flag_country']
        hover_data = list(set(df.columns) - set(columns_remove))
        # Sort by continent and city, both in ascending order
        df = df.sort_values(by=["continent", "country"], ascending=[True, True])
        # map with all cities
        maps.mapbox_map(df=df, hover_data=hover_data, hover_name="flag_city", file_name='mapbox_map_all')
        # Sort by continent and city, both in ascending order
        df = df.sort_values(by=["country", "city"], ascending=[True, True])

        # scatter plot for cities with number of videos over total time
        bivariate.scatter(df=df,
                          x="total_time",
                          y="video_count",
                          color="flag_country",
                          text="flag_city",
                          xaxis_title='Total time of footage (s)',
                          yaxis_title='Number of videos',
                          pretty_text=False,
                          marker_size=10,
                          save_file=True,
                          hover_data=hover_data,
                          hover_name="flag_city",
                          legend_title="",
                          # legend_x=0.01,
                          # legend_y=1.0,
                          label_distance_factor=5.0,
                          marginal_x=None,  # type: ignore
                          marginal_y=None,  # type: ignore
                          file_name='scatter_all_total_time-video_count')  # type: ignore
        # scatter plot for countries with number of videos over total time

        # compute total time per city first
        def safe_list_parse(s):
            """convert '[abc,def]' or '["abc","def"]' → ['abc','def']"""
            if not isinstance(s, str):
                return []
            s = s.strip()
            if s.startswith('[') and s.endswith(']'):
                # try to extract text chunks between commas, stripping quotes and spaces
                return [x.strip(" '\"") for x in re.split(r',\s*', s[1:-1]) if x.strip()]
            return []

        def safe_sum_parse(s):
            """convert nested '[ [123],[456] ]' → 123 + 456"""
            if not isinstance(s, str):
                return 0
            s = s.strip()
            if not s.startswith('['):
                return 0
            # extract all numbers (even nested ones)
            nums = re.findall(r'\d+', s)
            return sum(map(int, nums)) if nums else 0
        # compute totals per city
        df["city_video_count"] = df["videos"].apply(lambda x: len(safe_list_parse(x)))
        df["city_total_time"] = df["end_time"].apply(lambda x: safe_sum_parse(x))
        # aggregate to country level
        df_country = (
            df.groupby(["country", "iso3", "continent"], as_index=False)
              .agg(total_time=("city_total_time", "sum"),
                   video_count=("city_video_count", "sum"))
        )
        # add flag + iso3 label
        df_country["flag_country"] = df_country.apply(
            lambda row: f"{analysis_class.iso3_to_flag.get(row['iso3'], '🏳️')} {row['iso3']}",
            axis=1
        )
        # sort for readability
        df_country = df_country.sort_values(by=["continent", "country"], ascending=[True, True])
        # define hover data
        hover_data = ["country", "continent", "total_time", "video_count"]
        bivariate.scatter(df=df_country,
                          x="total_time",
                          y="video_count",
                          color="continent",
                          text="flag_country",
                          xaxis_title="Total time of footage (s)",
                          yaxis_title="Number of videos",
                          pretty_text=False,
                          marker_size=12,
                          save_file=True,
                          hover_data=hover_data,
                          hover_name="flag_country",
                          legend_title="",
                          label_distance_factor=0.1,
                          marginal_x=None,  # type: ignore
                          marginal_y=None,  # type: ignore
                          file_name="scatter_all_country_total_time-video_count")

        # histogram of dates of videos
        distribution.video_histogram_by_month(df=df,
                                              video_count_col='video_count',
                                              upload_date_col='upload_date',
                                              xaxis_title='Upload month (year-month)',
                                              yaxis_title='Number of videos',
                                              save_file=True)

        # maps with all cities and population heatmap
        maps.mapbox_map(df=df,
                        hover_data=hover_data,
                        density_col='population_city',
                        density_radius=10,
                        file_name='mapbox_map_all_pop')

        # maps with all cities and video count heatmap
        maps.mapbox_map(df=df,
                        hover_data=hover_data,
                        density_col='video_count',
                        density_radius=10,
                        file_name='mapbox_map_all_videos')

        # maps with all cities and total time heatmap
        maps.mapbox_map(df=df,
                        hover_data=hover_data,
                        density_col='total_time',
                        density_radius=10,
                        file_name='mapbox_map_all_time')

        # Type of vehicle over time of day
        df = df_mapping.copy()  # copy df to manipulate for output
        # --- expand rows so each video becomes one row ---
        expanded_rows = []
        for _, row in df.iterrows():
            try:
                vehicle_types = ast.literal_eval(row["vehicle_type"])
                times_of_day = ast.literal_eval(row["time_of_day"])
                if isinstance(vehicle_types, list) and isinstance(times_of_day, list):
                    for v_type, tod in zip(vehicle_types, times_of_day):
                        if not isinstance(v_type, list):
                            v_type = [v_type]
                        if not isinstance(tod, list):
                            tod = [tod]
                        for vt in v_type:
                            for t in tod:
                                expanded_rows.append({"vehicle_type": vt, "time_of_day": t})
            except Exception:
                pass  # skip malformed rows

        df_expanded = pd.DataFrame(expanded_rows)
        # --- map to human-readable labels ---
        df_expanded["vehicle_type_name"] = df_expanded["vehicle_type"].map(analysis_class.vehicle_map)
        df_expanded["time_of_day_name"] = df_expanded["time_of_day"].map(analysis_class.time_map)
        # drop rows where mapping failed
        df_expanded = df_expanded.dropna(subset=["vehicle_type_name", "time_of_day_name"])
        # --- aggregate counts ---
        df_summary = (
            df_expanded.groupby(["vehicle_type_name", "time_of_day_name"])
            .size()
            .reset_index(name="count")
        )
        # --- pivot into wide format for stacked bar plot ---
        df_pivot = (
            df_summary.pivot(index="vehicle_type_name", columns="time_of_day_name", values="count")
            .fillna(0)
            .reset_index()
        )
        # ensure consistent order of vehicle types
        vehicle_order = [
            "Car", "Bus", "Truck", "Two-wheeler", "Bicycle",
            "Automated car", "Automated bus", "Automated truck",
            "Automated two-wheeler", "Electric scooter"
        ]
        df_pivot["vehicle_type_name"] = pd.Categorical(df_pivot["vehicle_type_name"], categories=vehicle_order,
                                                       ordered=True)
        df_pivot = df_pivot.sort_values("vehicle_type_name")

        # --- plot ---
        distribution.bar(
            df=df_pivot,
            x=df_pivot["vehicle_type_name"],
            y=[col for col in ["Day", "Night"] if col in df_pivot.columns],
            y_legend=["Day", "Night"],
            stacked=True,
            pretty_text=False,
            orientation="v",
            xaxis_title="Type of vehicle",
            yaxis_title="Number of segments",
            show_text_labels=False,
            save_file=True,
            save_final=True,
            name_file="bar_vehicle_type_time_of_day"
        )

        # Continent over time of day
        df = df_mapping.copy()  # copy df to manipulate for output
        # --- expand rows so each video becomes one row ---
        expanded_rows = []
        for _, row in df.iterrows():
            try:
                times_of_day = ast.literal_eval(row["time_of_day"])
                if isinstance(times_of_day, list):
                    for tod in times_of_day:
                        if not isinstance(tod, list):
                            tod = [tod]
                        for t in tod:
                            expanded_rows.append({
                                "continent": row["continent"],
                                "time_of_day": t
                            })
            except Exception:
                pass  # skip malformed rows
        df_expanded = pd.DataFrame(expanded_rows)
        # --- map to human-readable labels ---
        df_expanded["time_of_day_name"] = df_expanded["time_of_day"].map(analysis_class.time_map)
        # drop rows where mapping failed
        df_expanded = df_expanded.dropna(subset=["time_of_day_name", "continent"])
        # --- aggregate counts ---
        df_summary = (
            df_expanded.groupby(["continent", "time_of_day_name"])
            .size()
            .reset_index(name="count")
        )
        # --- pivot into wide format for stacked bar plot ---
        df_pivot = (
            df_summary.pivot(index="continent", columns="time_of_day_name", values="count")
            .fillna(0)
            .reset_index()
        )
        # ensure only expected columns
        time_columns = [col for col in ["Day", "Night"] if col in df_pivot.columns]
        # --- plot ---
        distribution.bar(
            df=df_pivot,
            x=df_pivot["continent"],
            y=time_columns,
            y_legend=time_columns,
            stacked=True,
            pretty_text=False,
            orientation="v",
            xaxis_title="Continent",
            yaxis_title="Number of videos",
            show_text_labels=False,
            save_file=True,
            save_final=True,
            name_file="bar_continent_time_of_day"
        )

        total_duration = dataset_stats.calculate_total_seconds(df_mapping)

        # Displays values before applying filters
        logger.info(
            f"Duration of videos in seconds: {total_duration}, "
            f"in minutes: {total_duration/60:.2f}, "
            f"in hours: {total_duration/3600:.2f}, "
            f"in days: {total_duration/86400:.2f}, "
            f"in weeks: {total_duration/604800:.2f}, "
            f"in months: {total_duration/2629800:.2f}, "   # average month (30.44 days)
            f"in years: {total_duration/31557600:.2f} "    # average year (365.25 days)
            f"before filtering."
        )
        logger.info("Total number of videos before filtering: {}.",
                    dataset_stats.calculate_total_videos(df_mapping))

        country, number = metrics_cache.get_unique_values(df_mapping, "iso3")
        logger.info("Total number of countries and territories before filtering: {}.", number)

        city, number = metrics_cache.get_unique_values(df_mapping, "city")
        logger.info("Total number of cities before filtering: {}.", number)

        # Limit countries if required
        countries_include = common.get_configs("countries_analyse")
        if countries_include:
            df_mapping = df_mapping[df_mapping["iso3"].isin(common.get_configs("countries_analyse"))]

        # Make a dict for all columns
        city_country_cols = {
            # Object columns
            'person': 0, 'bicycle': 0, 'car': 0, 'motorcycle': 0, 'airplane': 0, 'bus': 0, 'train': 0,
            'truck': 0, 'boat': 0, 'traffic_light': 0, 'fire_hydrant': 0, 'stop_sign': 0, 'parking_meter': 0,
            'bench': 0, 'bird': 0, 'cat': 0, 'dog': 0, 'horse': 0, 'sheep': 0, 'cow': 0, 'elephant': 0, 'bear': 0,
            'zebra': 0, 'giraffe': 0, 'backpack': 0, 'umbrella': 0, 'handbag': 0, 'tie': 0, 'suitcase': 0,
            'frisbee': 0, 'skis': 0, 'snowboard': 0, 'sports_ball': 0, 'kite': 0, 'baseball_bat': 0,
            'baseball_glove': 0, 'skateboard': 0, 'surfboard': 0, 'tennis_racket': 0, 'bottle': 0, 'wine_glass': 0,
            'cup': 0, 'fork': 0, 'knife': 0, 'spoon': 0, 'bowl': 0, 'banana': 0, 'apple': 0, 'sandwich': 0,
            'orange': 0, 'broccoli': 0, 'carrot': 0, 'hot_dog': 0, 'pizza': 0, 'donut': 0, 'cake': 0, 'chair': 0,
            'couch': 0, 'potted_plant': 0, 'bed': 0, 'dining_table': 0, 'toilet': 0, 'tv': 0, 'laptop': 0,
            'mouse': 0, 'remote': 0, 'keyboard': 0, 'cellphone': 0, 'microwave': 0, 'oven': 0, 'toaster': 0,
            'sink': 0, 'refrigerator': 0, 'book': 0, 'clock': 0, 'vase': 0, 'scissors': 0, 'teddy_bear': 0,
            'hair_drier': 0, 'toothbrush': 0,

            'total_time': 0,
            'total_crossing_detect': 0,

            # City-level columns
            'speed_crossing_day_city': math.nan,
            'speed_crossing_night_city': math.nan,
            'speed_crossing_day_night_city_avg': math.nan,
            'time_crossing_day_city': math.nan,
            'time_crossing_night_city': math.nan,
            'time_crossing_day_night_city_avg': math.nan,
            'with_trf_light_day_city': 0.0,
            'with_trf_light_night_city': 0.0,
            'without_trf_light_day_city': 0.0,
            'without_trf_light_night_city': 0.0,
            'crossing_detected_city': 0,
            'crossing_detected_city_day': 0,
            'crossing_detected_city_night': 0,
            'crossing_detected_city_all': 0,
            'crossing_detected_city_all_day': 0,
            'crossing_detected_city_all_night': 0,

            # Country-level columns
            'speed_crossing_day_country': math.nan,
            'speed_crossing_night_country': math.nan,
            'speed_crossing_day_night_country_avg': math.nan,
            'time_crossing_day_country': math.nan,
            'time_crossing_night_country': math.nan,
            'time_crossing_day_night_country_avg': math.nan,
            'with_trf_light_day_country': 0.0,
            'with_trf_light_night_country': 0.0,
            'without_trf_light_day_country': 0.0,
            'without_trf_light_night_country': 0.0,
            'crossing_detected_country': 0,
            'crossing_detected_country_day': 0,
            'crossing_detected_country_night': 0,
            'crossing_detected_country_all': 0,
            'crossing_detected_country_all_day': 0,
            'crossing_detected_country_all_night': 0,
        }

        # Efficiently add all columns at once
        cols_df = pd.DataFrame([city_country_cols] * len(df_mapping), index=df_mapping.index)

        df_mapping = pd.concat([df_mapping, cols_df], axis=1)

        all_speed = {}
        all_time = {}

        logger.info("Processing csv files.")
        pedestrian_crossing_count, data = {}, {}
        pedestrian_crossing_count_all = {}

        for folder_path in common.get_configs("data"):  # Iterable[str]
            if not os.path.exists(folder_path):
                logger.warning(f"Folder does not exist: {folder_path}.")
                continue

            found_any = False

            for subfolder in common.get_configs("sub_domain"):
                subfolder_path = os.path.join(folder_path, subfolder)
                if not os.path.exists(subfolder_path):
                    continue

                found_any = True

                for file_name in tqdm(os.listdir(subfolder_path), desc=f"Processing files in {subfolder_path}"):
                    filtered: Optional[str] = analytics_IO.filter_csv_files(
                        file=file_name, df_mapping=df_mapping
                    )
                    if filtered is None:
                        continue

                    # Ensure "file" is always a string
                    file_str: str = os.fspath(filtered)  # converts PathLike to str safely

                    if file_str in MISC_FILES:
                        continue

                    filename_no_ext = os.path.splitext(file_str)[0]
                    logger.debug(f"{filename_no_ext}: fetching values.")

                    file_path = os.path.join(subfolder_path, file_str)
                    df = pd.read_csv(file_path)

                    # Keep only rows with confidence > min_conf
                    df = df[df["confidence"] >= common.get_configs("min_confidence")]

                    # After reading the file, clean up the filename
                    base_name = tools.clean_csv_filename(file_str)
                    filename_no_ext = os.path.splitext(base_name)[0]  # Remove extension

                    try:
                        video_id, start_index, fps = filename_no_ext.rsplit("_", 2)  # split to extract id and index
                    except ValueError:
                        logger.warning(f"Unexpected filename format: {filename_no_ext}")
                        continue

                    video_city_id = geo.find_city_id(df_mapping, video_id, int(start_index))
                    video_city = df_mapping.loc[df_mapping["id"] == video_city_id, "city"].values[0]  # type: ignore # noqa: E501
                    video_state = df_mapping.loc[df_mapping["id"] == video_city_id, "state"].values[0]  # type: ignore # noqa: E501
                    video_country = df_mapping.loc[df_mapping["id"] == video_city_id, "country"].values[0]  # type: ignore # noqa: E501
                    logger.debug(f"{file_str}: found values {video_city}, {video_state}, {video_country}.")

                    # Get the number of number and unique id of the object crossing the road
                    # ids give the unique of the person who cross the road after applying the filter, while
                    # all_ids gives every unique_id of the person who crosses the road
                    ids, all_ids = detection.pedestrian_crossing(df,
                                                                 filename_no_ext,
                                                                 df_mapping,
                                                                 common.get_configs("boundary_left"),
                                                                 common.get_configs("boundary_right"),
                                                                 person_id=0)

                    # Saving it in a dictionary in: {video-id_time: count, ids}
                    pedestrian_crossing_count[filename_no_ext] = {"ids": ids}
                    pedestrian_crossing_count_all[filename_no_ext] = {"ids": all_ids}

                    # Saves the time to cross in form {name_time: {id(s): time(s)}}
                    temp_data = metrics.time_to_cross(df,
                                                      pedestrian_crossing_count[filename_no_ext]["ids"],
                                                      filename_no_ext,
                                                      df_mapping)
                    data[filename_no_ext] = temp_data
                    # List of all 80 class names in COCO order
                    coco_classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
                                    'boat', 'traffic_light', 'fire_hydrant', 'stop_sign', 'parking_meter', 'bench',
                                    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
                                    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                                    'skis', 'snowboard', 'sports_ball', 'kite', 'baseball_bat', 'baseball_glove',
                                    'skateboard', 'surfboard', 'tennis_racket', 'bottle', 'wine_glass', 'cup',
                                    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
                                    'broccoli', 'carrot', 'hot_dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                                    'potted_plant', 'bed', 'dining_table', 'toilet', 'tv', 'laptop', 'mouse',
                                    'remote', 'keyboard', 'cellphone', 'microwave', 'oven', 'toaster', 'sink',
                                    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy_bear',
                                    'hair_drier', 'toothbrush']

                    # --- Ensure all needed columns exist and are integer type ---
                    for class_name in coco_classes:
                        if class_name not in df_mapping.columns:
                            df_mapping[class_name] = 0
                        df_mapping[class_name] = pd.to_numeric(df_mapping[class_name],
                                                               errors='coerce').fillna(0).astype(int)  # type: ignore
                    # --- Count unique objects per yolo-id ---
                    object_counts = (
                        df.drop_duplicates(['yolo-id', 'unique-id'])['yolo-id']
                        .value_counts().sort_index()
                    )
                    counters = {class_name: int(object_counts.get(i, 0)) for i,
                                class_name in enumerate(coco_classes)}

                    # --- Update df_mapping for the given video_city_id ---
                    for class_name in coco_classes:
                        df_mapping.loc[df_mapping["id"] == video_city_id, class_name] += counters[class_name]  # type: ignore  # noqa: E501

                    # Add duration of segment
                    time_video = duration.get_duration(df_mapping, video_id, int(start_index))
                    df_mapping.loc[df_mapping["id"] == video_city_id, "total_time"] += time_video  # type: ignore

                    # Add total crossing detected
                    df_mapping.loc[df_mapping["id"] == video_city_id, "total_crossing_detect"] += len(ids)  # type: ignore  # noqa: E501

                    # Aggregated values
                    speed_value = metrics.calculate_speed_of_crossing(df_mapping,
                                                                      df,
                                                                      {filename_no_ext: temp_data})
                    if speed_value is not None:
                        for outer_key, inner_dict in speed_value.items():
                            if outer_key not in all_speed:
                                all_speed[outer_key] = inner_dict
                            else:
                                all_speed[outer_key].update(inner_dict)
                    time_value = metrics.time_to_start_cross(df_mapping,
                                                             df,
                                                             {filename_no_ext: temp_data})
                    if time_value is not None:
                        for outer_key, inner_dict in time_value.items():
                            if outer_key not in all_time:
                                all_time[outer_key] = inner_dict
                            else:
                                all_time[outer_key].update(inner_dict)

        person_counter = df_mapping['person'].sum()
        bicycle_counter = df_mapping['bicycle'].sum()
        car_counter = df_mapping['car'].sum()
        motorcycle_counter = df_mapping['motorcycle'].sum()
        bus_counter = df_mapping['bus'].sum()
        truck_counter = df_mapping['truck'].sum()
        cellphone_counter = df_mapping['cellphone'].sum()
        traffic_light_counter = df_mapping['traffic_light'].sum()
        stop_sign_counter = df_mapping['stop_sign'].sum()

        # Record the average speed and time of crossing on country basis
        avg_speed_country, all_speed_country = metrics.avg_speed_of_crossing_country(df_mapping, all_speed)

        # Output in real world seconds
        avg_time_country, all_time_country = metrics.avg_time_to_start_cross_country(df_mapping, all_time)

        # Record the average speed and time of crossing on city basis
        avg_speed_city, all_speed_city = metrics.avg_speed_of_crossing_city(df_mapping, all_speed)
        avg_time_city, all_time_city = metrics.avg_time_to_start_cross_city(df_mapping, all_time)

        # Kill the program if there is no data to analyse
        if len(avg_time_city) == 0 or len(avg_speed_city) == 0:
            logger.error("No speed and time data to analyse.")
            exit()

        logger.info("Calculating counts of detected traffic signs.")
        traffic_sign_city = metrics_cache.calculate_traffic_signs(df_mapping)

        logger.info("Calculating counts of detected mobile phones.")
        cellphone_city = metrics_cache.calculate_cellphones(df_mapping)

        logger.info("Calculating counts of detected vehicles.")
        vehicle_city = metrics_cache.calculate_traffic(df_mapping, motorcycle=1, car=1, bus=1, truck=1)

        logger.info("Calculating counts of detected bicycles.")
        bicycle_city = metrics_cache.calculate_traffic(df_mapping, bicycle=1)

        logger.info("Calculating counts of detected cars (subset of vehicles).")
        car_city = metrics_cache.calculate_traffic(df_mapping, car=1)

        logger.info("Calculating counts of detected motorcycles (subset of vehicles).")
        motorcycle_city = metrics_cache.calculate_traffic(df_mapping, motorcycle=1)

        logger.info("Calculating counts of detected buses (subset of vehicles).")
        bus_city = metrics_cache.calculate_traffic(df_mapping, bus=1)

        logger.info("Calculating counts of detected trucks (subset of vehicles).")
        truck_city = metrics_cache.calculate_traffic(df_mapping, truck=1)

        logger.info("Calculating counts of detected persons.")
        person_city = metrics_cache.calculate_traffic(df_mapping, person=1)

        logger.info("Calculating counts of detected crossing events with traffic lights.")
        cross_evnt_city = events.crossing_event_wt_traffic_light(df_mapping, data)

        logger.info("Calculating counts of crossing events in cities.")
        pedestrian_cross_city = dataset_stats.pedestrian_cross_per_city(pedestrian_crossing_count, df_mapping)
        pedestrian_cross_city_all = dataset_stats.pedestrian_cross_per_city(pedestrian_crossing_count_all,
                                                                            df_mapping)

        logger.info("Calculating counts of crossing events in countries.")
        pedestrian_cross_country = dataset_stats.pedestrian_cross_per_country(pedestrian_cross_city, df_mapping)
        pedestrian_cross_country_all = dataset_stats.pedestrian_cross_per_country(pedestrian_cross_city_all,
                                                                                  df_mapping)

        # Jaywalking data
        logger.info("Calculating parameters for detection of jaywalking.")

        (crossings_with_traffic_equipment_city, crossings_without_traffic_equipment_city,
         total_duration_by_city, crossings_with_traffic_equipment_country, crossings_without_traffic_equipment_country,
         total_duration_by_country) = events.crossing_event_with_traffic_equipment(df_mapping, data)

        # ----------------------------------------------------------------------
        # Add city-level crossing counts for with and without traffic equipment
        # ----------------------------------------------------------------------

        for key, value in crossings_with_traffic_equipment_city.items():
            parts = key.split("_")
            city = parts[0]
            lat = parts[1]
            long = parts[2]
            time_of_day = int(parts[3])  # 0 = day, 1 = night

            # Optional: Extract state if available
            state = df_mapping.loc[df_mapping["city"] == city,
                                   "state"].iloc[0] if "state" in df_mapping.columns else None  # type: ignore

            colname = "with_trf_light_day_city" if not time_of_day else "with_trf_light_night_city"

            df_mapping.loc[
                (df_mapping["city"] == city) &
                ((df_mapping["state"] == state) | (pd.isna(df_mapping["state"]) & pd.isna(state))),
                colname
            ] = float(value)

        for key, value in crossings_without_traffic_equipment_city.items():
            parts = key.split("_")
            city = parts[0]
            lat = parts[1]
            long = parts[2]
            time_of_day = int(parts[3])

            # Optional: Extract state if available
            state = df_mapping.loc[df_mapping["city"] == city,
                                   "state"].iloc[0] if "state" in df_mapping.columns else None  # type: ignore

            colname = "without_trf_light_day_city" if not time_of_day else "without_trf_light_night_city"

            df_mapping.loc[
                (df_mapping["city"] == city) &
                ((df_mapping["state"] == state) | (pd.isna(df_mapping["state"]) & pd.isna(state))),
                colname
            ] = float(value)

        # ----------------------------------------------------------------------
        # Add country-level crossing counts for with and without traffic equipment
        # ----------------------------------------------------------------------

        for key, value in crossings_with_traffic_equipment_country.items():
            parts = key.split("_")
            country = parts[0]
            time_of_day = int(parts[1])

            colname = "with_trf_light_day_country" if not time_of_day else "with_trf_light_night_country"

            df_mapping.loc[
                (df_mapping["country"] == country),
                colname
            ] = float(value)

        for key, value in crossings_without_traffic_equipment_country.items():
            parts = key.split("_")
            country = parts[0]
            time_of_day = int(parts[1])

            colname = "without_trf_light_day_country" if not time_of_day else "without_trf_light_night_country"

            df_mapping.loc[
                (df_mapping["country"] == country),
                colname
            ] = float(value)

        # ---------------------------------------
        # Add city-level crossing counts detected
        # ---------------------------------------
        for city_long_lat_cond, value in pedestrian_cross_city.items():
            city, lat, long, cond = city_long_lat_cond.split('_')
            lat = float(lat)  # lat column is float

            # Set the correct column name based on condition
            if cond == "0":
                target_column = "crossing_detected_city_day"
            elif cond == "1":
                target_column = "crossing_detected_city_night"
            else:
                continue  # skip if cond is not recognised

            # Set the value in the right place
            df_mapping.loc[
                (df_mapping["city"] == city) & (df_mapping["lat"] == lat),
                target_column
            ] = float(value)

        for city_long_lat_cond, value in pedestrian_cross_city_all.items():
            city, lat, long, cond = city_long_lat_cond.split('_')
            lat = float(lat)  # if your lat column is float
            # Set the correct column name based on condition
            if cond == "0":
                target_column = "crossing_detected_city_all_day"
            elif cond == "1":
                target_column = "crossing_detected_city_all_night"
            else:
                continue  # skip if cond is not recognized
            # Set the value in the right place
            df_mapping.loc[
                (df_mapping["city"] == city) & (df_mapping["lat"] == lat),
                target_column
            ] = float(value)

        df_mapping["crossing_detected_city"] = (
            df_mapping["crossing_detected_city_day"].fillna(0)
            + df_mapping["crossing_detected_city_night"].fillna(0)
        )

        df_mapping["crossing_detected_city_all"] = (
            df_mapping["crossing_detected_city_all_day"].fillna(0)
            + df_mapping["crossing_detected_city_all_night"].fillna(0)
        )

        # ---------------------------------------
        # Add country-level crossing counts detected
        # ---------------------------------------
        for country_cond, value in pedestrian_cross_country.items():
            country, cond = country_cond.split('_')
            # Set the correct column name based on condition
            if cond == "0":
                target_column = "crossing_detected_country_day"
            elif cond == "1":
                target_column = "crossing_detected_country_night"
            else:
                continue  # skip if cond is not recognized
            # Set the value in the right place
            df_mapping.loc[
                (df_mapping["country"] == country),
                target_column
            ] = float(value)

        for country_cond, value in pedestrian_cross_country_all.items():
            country, cond = country_cond.split('_')
            # Set the correct column name based on condition
            if cond == "0":
                target_column = "crossing_detected_country_all_day"
            elif cond == "1":
                target_column = "crossing_detected_country_all_night"
            else:
                continue  # skip if cond is not recognized
            # Set the value in the right place
            df_mapping.loc[
                (df_mapping["country"] == country),
                target_column
            ] = float(value)

        df_mapping["crossing_detected_country"] = (
            df_mapping["crossing_detected_country_day"].fillna(0)
            + df_mapping["crossing_detected_country_night"].fillna(0)
        )

        df_mapping["crossing_detected_country_all"] = (
            df_mapping["crossing_detected_country_all_day"].fillna(0)
            + df_mapping["crossing_detected_country_all_night"].fillna(0)
        )

        # Add column with count of videos
        df_mapping["total_videos"] = df_mapping["videos"].apply(lambda x: len(x.strip("[]").split(",")) if x.strip("[]") else 0)  # noqa: E501

        # Get lat and lon for cities
        logger.info("Fetching lat and lon coordinates for cities.")
        for index, row in tqdm(df_mapping.iterrows(), total=len(df_mapping)):
            if pd.isna(row["lat"]) or pd.isna(row["lon"]):
                lat, lon = geo.get_coordinates(row["city"],
                                               row["state"],
                                               common.correct_country(row["country"]))  # type: ignore

                df_mapping.at[index, 'lat'] = lat  # type: ignore
                df_mapping.at[index, 'lon'] = lon  # type: ignore

        # Save the raw file for further investigation
        df_mapping_raw = df_mapping.copy()

        df_mapping_raw.drop(['gmp', 'population_city', 'population_country', 'traffic_mortality',
                             'literacy_rate', 'avg_height', 'med_age', 'gini', 'traffic_index', 'videos',
                             'time_of_day', 'start_time', 'end_time', 'vehicle_type', 'upload_date',
                             ], axis=1, inplace=True)

        df_mapping_raw['channel'] = df_mapping_raw['channel'].apply(tools.count_unique_channels)
        df_mapping_raw.to_csv(os.path.join(common.output_dir, "mapping_city_raw.csv"))

        # Get the population threshold from the configuration
        population_threshold = common.get_configs("population_threshold")

        # Get the minimum percentage of country population from the configuration
        min_percentage = common.get_configs("min_city_population_percentage")

        # Convert 'population_city' to numeric (force errors to NaN)
        df_mapping["population_city"] = pd.to_numeric(df_mapping["population_city"], errors='coerce')

        # Filter df_mapping to include cities that meet either of the following criteria:
        # 1. The city's population is greater than the threshold
        # 2. The city's population is at least the minimum percentage of the country's population
        df_mapping = df_mapping[
            (df_mapping["population_city"] >= population_threshold) |  # Condition 1
            (df_mapping["population_city"] >= min_percentage * df_mapping["population_country"])  # Condition 2
        ]

        # Remove the rows of the cities where the footage recorded is less than threshold
        df_mapping = dataset_stats.remove_columns_below_threshold(df_mapping, common.get_configs("footage_threshold"))

        # Limit countries if required
        countries_include = common.get_configs("countries_analyse")
        if countries_include:
            df_mapping = df_mapping[df_mapping["iso3"].isin(common.get_configs("countries_analyse"))]

        total_duration = dataset_stats.calculate_total_seconds(df_mapping)

        # Displays values after applying filters
        logger.info(f"Duration of videos in seconds after filtering: {total_duration}, in" +
                    f" minutes after filtering: {total_duration/60:.2f}, in " +
                    f"hours: {total_duration/60/60:.2f}.")

        logger.info("Total number of videos after filtering: {}.",
                    dataset_stats.calculate_total_videos(df_mapping))

        country, number = metrics_cache.get_unique_values(df_mapping, "iso3")
        logger.info("Total number of countries and territories after filtering: {}.", number)

        city, number = metrics_cache.get_unique_values(df_mapping, "city")
        logger.info("Total number of cities after filtering: {}.", number)

        df_mapping = mapping_enrich.add_speed_and_time_to_mapping(df_mapping=df_mapping,
                                                                  avg_speed_city=avg_speed_city,
                                                                  avg_speed_country=avg_speed_country,
                                                                  avg_time_city=avg_time_city,
                                                                  avg_time_country=avg_time_country,
                                                                  pedestrian_cross_city=pedestrian_cross_city,
                                                                  pedestrian_cross_country=pedestrian_cross_country)

        min_max_speed = duration.get_duration_segment(all_speed, df_mapping, name="speed", duration=None)
        min_max_time = duration.get_duration_segment(all_time, df_mapping, name="time", duration=None)

        # Save the results to a pickle file
        logger.info("Saving results to a pickle file {}.", file_results)
        with open(file_results, 'wb') as file:
            pickle.dump((data,                                              # 0
                         person_counter,                                    # 1
                         bicycle_counter,                                   # 2
                         car_counter,                                       # 3
                         motorcycle_counter,                                # 4
                         bus_counter,                                       # 5
                         truck_counter,                                     # 6
                         cellphone_counter,                                 # 7
                         traffic_light_counter,                             # 8
                         stop_sign_counter,                                 # 9
                         pedestrian_cross_city,                             # 10
                         pedestrian_crossing_count,                         # 11
                         person_city,                                       # 12
                         bicycle_city,                                      # 13
                         car_city,                                          # 14
                         motorcycle_city,                                   # 15
                         bus_city,                                          # 16
                         truck_city,                                        # 17
                         cross_evnt_city,                                   # 18
                         vehicle_city,                                      # 19
                         cellphone_city,                                    # 20
                         traffic_sign_city,                                 # 21
                         all_speed,                                         # 22
                         all_time,                                          # 23
                         avg_time_city,                                     # 24
                         avg_speed_city,                                    # 25
                         df_mapping,                                        # 26
                         avg_speed_country,                                 # 27
                         avg_time_country,                                  # 28
                         crossings_with_traffic_equipment_city,             # 29
                         crossings_without_traffic_equipment_city,          # 30
                         crossings_with_traffic_equipment_country,          # 31
                         crossings_without_traffic_equipment_country,       # 32
                         min_max_speed,                                     # 33
                         min_max_time,                                      # 34
                         pedestrian_cross_country,                          # 35
                         all_speed_city,                                    # 36
                         all_time_city,                                     # 37
                         all_speed_country,                                 # 38
                         all_time_country,                                  # 39
                         df_mapping_raw,                                    # 40
                         pedestrian_cross_city_all,                         # 41
                         pedestrian_cross_country_all),                     # 42
                        file)

        logger.info("Analysis results saved to pickle file.")

    # Set index as ID
    df_mapping = df_mapping.set_index("id", drop=False)

    # --- Check if reanalysis of speed is required ---
    if common.get_configs("reanalyse_speed"):
        # Compute average speed for each country using mapping and speed data
        avg_speed_country = metrics.avg_speed_of_crossing_country(df_mapping, all_speed)
        # Compute average speed for each city using speed data
        avg_speed_city = metrics.avg_speed_of_crossing_city(df_mapping, all_speed)

        # Add computed speed values to the main mapping dataframe
        df_mapping = mapping_enrich.add_speed_and_time_to_mapping(
            df_mapping=df_mapping,
            avg_speed_city=avg_speed_city,
            avg_time_city=None,
            avg_speed_country=avg_speed_country,
            avg_time_country=None,
            pedestrian_cross_city=pedestrian_cross_city,
            pedestrian_cross_country=pedestrian_cross_country
        )

        # --- Update avg speed values in the pickle file ---
        with open(file_results, 'rb') as file:
            results = pickle.load(file)  # Load existing results

        results_list = list(results)
        results_list[25] = avg_speed_city     # Update city speed
        results_list[27] = avg_speed_country  # Update country speed
        results_list[26] = df_mapping         # Update mapping

        with open(file_results, 'wb') as file:
            pickle.dump(tuple(results_list), file)  # Save updated results
        logger.info("Updated speed values in the pickle file.")

    # --- Check if reanalysis of waiting time is required ---
    if common.get_configs("reanalyse_waiting_time"):
        # Compute average waiting time to start crossing for each country
        avg_time_country = metrics.avg_time_to_start_cross_country(df_mapping, all_speed)
        # Compute average waiting time to start crossing for each city
        avg_time_city = metrics.avg_time_to_start_cross_city(df_mapping, all_time)

        # Add computed time values to the main mapping dataframe
        df_mapping = mapping_enrich.add_speed_and_time_to_mapping(
            df_mapping=df_mapping,
            avg_time_city=avg_time_city,
            avg_speed_city=avg_speed_city,
            avg_time_country=avg_time_country,
            avg_speed_country=avg_speed_country,
            pedestrian_cross_city=pedestrian_cross_city,
            pedestrian_cross_country=pedestrian_cross_country
        )

        # --- Update avg time values in the pickle file ---
        with open(file_results, 'rb') as file:
            results = pickle.load(file)  # Load existing results

        results_list = list(results)
        results_list[24] = avg_time_city     # Update city waiting time
        results_list[28] = avg_time_country  # Update country waiting time
        results_list[26] = df_mapping        # Update mapping

        with open(file_results, 'wb') as file:
            pickle.dump(tuple(results_list), file)  # Save updated results
        logger.info("Updated time values in the pickle file.")

    # --- Remove countries/cities with insufficient crossing detections ---
    if common.get_configs("min_crossing_detect") != 0:
        # Group values by country
        threshold: float = float(common.get_configs("min_crossing_detect"))
        country_detect: Dict[str, Dict[str, float]] = {}
        for key, value in pedestrian_cross_country.items():
            country, cond = key.rsplit('_', 1)
            val_f = float(value)
            if country not in country_detect:
                country_detect[country] = {}
            country_detect[country][cond] = val_f

        # Find countries where BOTH conditions are below threshold
        keep_countries: Set[str] = {
            country for country, vals in country_detect.items()
            if (('0' in vals or '1' in vals) and
                (vals.get('0', 0.0) + vals.get('1', 0.0) >= threshold))
        }

        df_mapping = df_mapping[df_mapping['country'].isin(keep_countries)].copy()
        # # Remove all entries in avg_speed_country and avg_time_country for those countries
        # for dict_name, d in [('avg_speed_country', avg_speed_country), ('avg_time_country', avg_time_country)]:
        #     keys_to_remove = [key for key in d if key.split('_')[0] in remove_countries]  # type: ignore
        #     for key in keys_to_remove:
        #         logger.debug(f"Deleting from {dict_name}: {key} -> {d[key]}")  # type: ignore
        #         del d[key]  # type: ignore

        # --- Remove low-detection cities from city-level speed/time ---
        # Sum all conditions for each city in pedestrian_cross_city
        # city_sum = defaultdict(int)
        # for key, value in pedestrian_cross_city.items():
        #     city = key.split('_')[0]
        #     city_sum[city] += value

        # Find cities with total crossings below threshold
        # remove_cities = {city for city, total in city_sum.items()
        #                  if total < common.get_configs("min_crossing_detect")}

        # # Remove rows from df_mapping where 'cities' is in remove_cities
        # df_mapping = df_mapping[~df_mapping['city'].isin(remove_cities)].copy()

        # # Remove all entries in avg_speed_city and avg_time_city for those cities
        # for dict_name, d in [('avg_speed_city', avg_speed_city), ('avg_time_city', avg_time_city)]:
        #     keys_to_remove = [key for key in d if key.split('_')[0] in remove_cities]  # type: ignore
        #     for key in keys_to_remove:
        #         logger.debug(f"Deleting from {dict_name}: {key} -> {d[key]}")  # type: ignore
        #         del d[key]  # type: ignore

    # Sort by continent and city, both in ascending order
    df_mapping = df_mapping.sort_values(by=["continent", "city"], ascending=[True, True])

    # Save updated mapping file in output
    os.makedirs(common.output_dir, exist_ok=True)  # check if folder
    df_mapping.to_csv(os.path.join(common.output_dir, "mapping_updated.csv"))

    logger.info("Detected:")
    logger.info(f"person: {person_counter}; bicycle: {bicycle_counter}; car: {car_counter}")
    logger.info(f"motorcycle: {motorcycle_counter}; bus: {bus_counter}; truck: {truck_counter}")
    logger.info(f"cellphone: {cellphone_counter}; traffic light: {traffic_light_counter}; " +
                f"traffic sign: {stop_sign_counter}")

    logger.info("Producing output.")

    # Data to avoid showing on hover in scatter plots
    columns_remove = ['videos', 'time_of_day', 'start_time', 'end_time', 'upload_date', 'fps_list', 'vehicle_type',
                      'channel']
    hover_data = list(set(df_mapping.columns) - set(columns_remove))

    df = df_mapping.copy()  # copy df to manipulate for output
    df['state'] = df['state'].fillna('NA')  # Set state to NA

    # Maps with filtered data
    maps.mapbox_map(df=df, hover_data=hover_data, file_name='mapbox_map')

    maps.mapbox_map(df=df,
                    hover_data=hover_data,
                    density_col='total_time',
                    density_radius=10,
                    file_name='mapbox_map_time')

    maps.world_map(df_mapping=df)  # map with countries

    distribution.violin_plot(data_index=22,
                             name="speed",
                             min_threshold=common.get_configs("min_speed_limit"),
                             max_threshold=common.get_configs("max_speed_limit"), df_mapping=df_mapping,
                             save_file=True, data_file=file_results)

    # ------------All values----------------- #
    distribution.hist(data_index=22,
                      name="speed",
                      marginal="violin",
                      nbins=100,
                      raw=True,
                      min_threshold=common.get_configs("min_speed_limit"),
                      max_threshold=common.get_configs("max_speed_limit"),
                      font_size=common.get_configs("font_size") + 4,
                      fig_save_height=650,
                      save_file=True,
                      data_file=file_results)

    distribution.hist(data_index=39,
                      name="time",
                      marginal="violin",
                      # nbins=100,
                      raw=True,
                      min_threshold=None,
                      max_threshold=None,
                      font_size=common.get_configs("font_size") + 4,
                      fig_save_height=650,
                      save_file=True,
                      data_file=file_results)

    # ------------Filtered values----------------- #
    distribution.hist(data_index=38,
                      name="speed_filtered",
                      marginal="violin",
                      nbins=100,
                      raw=False,
                      min_threshold=common.get_configs("min_speed_limit"),
                      max_threshold=common.get_configs("max_speed_limit"),
                      font_size=common.get_configs("font_size") + 4,
                      fig_save_height=650,
                      save_file=True,
                      data_file=file_results)

    distribution.hist(data_index=37,
                      name="time_filtered",
                      marginal="violin",
                      # nbins=100,
                      raw=False,
                      min_threshold=None,
                      max_threshold=None,
                      font_size=common.get_configs("font_size") + 4,
                      df_mapping=df_mapping,
                      fig_save_height=650,
                      save_file=True,
                      data_file=file_results)

    if common.get_configs("analysis_level") == "city":

        # Amount of footage
        bivariate.scatter(df=df,
                          x="total_time",
                          y="person",
                          color="continent",
                          text="city",
                          xaxis_title='Total time of footage (s)',
                          yaxis_title='Number of detected pedestrians',
                          pretty_text=False,
                          marker_size=10,
                          save_file=True,
                          hover_data=hover_data,
                          hover_name="city",
                          legend_title="",
                          legend_x=0.01,
                          legend_y=1.0,
                          label_distance_factor=5.0,
                          marginal_x=None,  # type: ignore
                          marginal_y=None)  # type: ignore

        # todo: ISO-3 codes next to figures shift. need to correct once "final" dataset is online
        crossing.speed_and_time_to_start_cross(df_mapping,
                                               x_axis_title_height=110,
                                               font_size_captions=common.get_configs("font_size") + 8,
                                               legend_x=0.9,
                                               legend_y=0.01,
                                               legend_spacing=0.0026)

        stacked.stack_plot(df,
                           order_by="alphabetical",
                           metric="time",
                           data_view="combined",
                           title_text="Crossing initiation time (s)",
                           filename="time_crossing_alphabetical",
                           font_size_captions=common.get_configs("font_size") + 8,
                           left_margin=80,
                           right_margin=80
                           )

        stacked.stack_plot(df,
                           order_by="alphabetical",
                           metric="time",
                           data_view="day",
                           title_text="Crossing initiation time (s)",
                           filename="time_crossing_alphabetical_day",
                           font_size_captions=common.get_configs("font_size") + 8,
                           left_margin=80,
                           right_margin=80
                           )

        stacked.stack_plot(df,
                           order_by="alphabetical",
                           metric="time",
                           data_view="night",
                           title_text="Crossing initiation time (s)",
                           filename="time_crossing_alphabetical_night",
                           font_size_captions=common.get_configs("font_size") + 8,
                           left_margin=80,
                           right_margin=80
                           )

        stacked.stack_plot(df,
                           order_by="average",
                           metric="time",
                           data_view="combined",
                           title_text="Crossing initiation time (s)",
                           filename="time_crossing_avg",
                           font_size_captions=common.get_configs("font_size") + 8,
                           left_margin=10,
                           right_margin=10
                           )

        stacked.stack_plot(df,
                           order_by="average",
                           metric="time",
                           data_view="day",
                           title_text="Crossing initiation time (s)",
                           filename="time_crossing_avg_day",
                           font_size_captions=common.get_configs("font_size") + 8,
                           left_margin=10,
                           right_margin=10
                           )

        stacked.stack_plot(df,
                           order_by="average",
                           metric="time",
                           data_view="night",
                           title_text="Crossing initiation time (s)",
                           filename="time_crossing_avg_night",
                           font_size_captions=common.get_configs("font_size") + 8,
                           left_margin=10,
                           right_margin=10
                           )

        stacked.stack_plot(df,
                           order_by="alphabetical",
                           metric="speed",
                           data_view="combined",
                           title_text="Mean speed of crossing (in m/s)",
                           filename="speed_crossing_alphabetical",
                           font_size_captions=common.get_configs("font_size") + 8,
                           left_margin=80,
                           right_margin=80
                           )

        stacked.stack_plot(df,
                           order_by="alphabetical",
                           metric="speed",
                           data_view="day",
                           title_text="Mean speed of crossing (in m/s)",
                           filename="speed_crossing_alphabetical_day",
                           font_size_captions=common.get_configs("font_size") + 8,
                           left_margin=80,
                           right_margin=80
                           )

        stacked.stack_plot(df,
                           order_by="alphabetical",
                           metric="speed",
                           data_view="night",
                           title_text="Mean speed of crossing (in m/s)",
                           filename="speed_crossing_alphabetical_night",
                           font_size_captions=common.get_configs("font_size") + 8,
                           left_margin=80,
                           right_margin=80
                           )

        stacked.stack_plot(df,
                           order_by="average",
                           metric="speed",
                           data_view="combined",
                           title_text="Mean speed of crossing (in m/s)",
                           filename="speed_crossing_avg",
                           font_size_captions=common.get_configs("font_size") + 8,
                           left_margin=10,
                           right_margin=10
                           )

        stacked.stack_plot(df,
                           order_by="average",
                           metric="speed",
                           data_view="day",
                           title_text="Mean speed of crossing (in m/s)",
                           filename="speed_crossing_avg_day",
                           font_size_captions=common.get_configs("font_size") + 8,
                           left_margin=10,
                           right_margin=10
                           )

        stacked.stack_plot(df,
                           order_by="average",
                           metric="speed",
                           data_view="night",
                           title_text="Mean speed of crossing (in m/s)",
                           filename="speed_crossing_avg_night",
                           font_size_captions=common.get_configs("font_size") + 8,
                           left_margin=10,
                           right_margin=10
                           )

        correlation.correlation_matrix(df_mapping, pedestrian_cross_city, person_city, bicycle_city,
                                       car_city, motorcycle_city, bus_city, truck_city, cross_evnt_city,
                                       vehicle_city, cellphone_city, traffic_sign_city, all_speed, all_time,
                                       avg_time_city, avg_speed_city)

        # Speed of crossing vs time to start crossing
        df = df_mapping[df_mapping["speed_crossing"] != 0].copy()
        df = df[df["time_crossing"] != 0]
        df['state'] = df['state'].fillna('NA')
        bivariate.scatter(df=df,
                          x="speed_crossing",
                          y="time_crossing",
                          color="continent",
                          text="city",
                          xaxis_title='Speed of crossing (in m/s)',
                          yaxis_title='Crossing initiation time (in s)',
                          pretty_text=False,
                          marker_size=10,
                          save_file=True,
                          hover_data=hover_data,
                          hover_name="city",
                          legend_title="",
                          legend_x=0.01,
                          legend_y=1.0,
                          label_distance_factor=3.0,
                          marginal_x=None,  # type: ignore
                          marginal_y=None)  # type: ignore

        # Speed of crossing during daytime vs time to start crossing during daytime
        df = df_mapping[df_mapping["speed_crossing_day"] != 0].copy()
        df = df[df["time_crossing_day"] != 0]
        df['state'] = df['state'].fillna('NA')

        bivariate.scatter(df=df,
                          x="speed_crossing_day",
                          y="time_crossing_day",
                          color="continent",
                          text="city",
                          xaxis_title='Crossing speed during daytime (in m/s)',
                          yaxis_title='Crossing initiation time during daytime (in s)',
                          pretty_text=False,
                          marker_size=10,
                          save_file=True,
                          hover_data=hover_data,
                          hover_name="city",
                          legend_title="",
                          legend_x=0.01,
                          legend_y=1.0,
                          label_distance_factor=3.0,
                          marginal_x=None,  # type: ignore
                          marginal_y=None)  # type: ignore

        # Speed of crossing during night time vs time to start crossing during night time
        df = df_mapping[df_mapping["speed_crossing_night"] != 0].copy()
        df = df[df["time_crossing_night"] != 0]
        df['state'] = df['state'].fillna('NA')
        bivariate.scatter(df=df,
                          x="speed_crossing_night",
                          y="time_crossing_night",
                          color="continent",
                          text="city",
                          xaxis_title='Crossing speed during night time (in m/s)',
                          yaxis_title='Crossing initiation time during night time (in s)',
                          pretty_text=False,
                          marker_size=10,
                          save_file=True,
                          hover_data=hover_data,
                          hover_name="city",
                          legend_title="",
                          legend_x=0.87,
                          legend_y=1.0,
                          label_distance_factor=0.8,
                          marginal_x=None,  # type: ignore
                          marginal_y=None)  # type: ignore

        # Time to start crossing vs population of city
        df = df_mapping[df_mapping["time_crossing"] != 0].copy()
        df = df[(df["population_city"].notna()) & (df["population_city"] != 0)]
        df['state'] = df['state'].fillna('NA')
        bivariate.scatter(df=df,
                          x="time_crossing",
                          y="population_city",
                          color="continent",
                          text="city",
                          xaxis_title='Crossing initiation time (in s)',
                          yaxis_title='Population of city',
                          pretty_text=False,
                          marker_size=10,
                          save_file=True,
                          hover_data=hover_data,
                          hover_name="city",
                          legend_title="",
                          legend_x=0.87,
                          legend_y=1.0,
                          label_distance_factor=2.0,
                          marginal_x=None,  # type: ignore
                          marginal_y=None)  # type: ignore

        # Speed of crossing vs population of city
        df = df_mapping[df_mapping["speed_crossing"] != 0].copy()
        df = df[(df["population_city"].notna()) & (df["population_city"] != 0)]
        df['state'] = df['state'].fillna('NA')
        bivariate.scatter(df=df,
                          x="speed_crossing",
                          y="population_city",
                          color="continent",
                          text="city",
                          xaxis_title='Mean speed of crossing (in m/s)',
                          yaxis_title='Population of city',
                          pretty_text=False,
                          marker_size=10,
                          save_file=True,
                          hover_data=hover_data,
                          hover_name="city",
                          legend_title="",
                          legend_x=0.87,
                          legend_y=1.0,
                          label_distance_factor=3.0,
                          marginal_x=None,  # type: ignore
                          marginal_y=None)  # type: ignore

        # Time to start crossing vs population of city
        df = df_mapping[df_mapping["time_crossing"] != 0].copy()
        df = df[(df["traffic_mortality"].notna()) & (df["traffic_mortality"] != 0)]
        df['state'] = df['state'].fillna('NA')
        bivariate.scatter(df=df,
                          x="time_crossing",
                          y="traffic_mortality",
                          color="continent",
                          text="city",
                          xaxis_title='Crossing initiation time (in s)',
                          yaxis_title='National traffic mortality rate (per 100,000 of population)',
                          pretty_text=False,
                          marker_size=10,
                          save_file=True,
                          hover_data=hover_data,
                          hover_name="city",
                          legend_title="",
                          legend_x=0.87,
                          legend_y=1.0,
                          label_distance_factor=3.0,
                          marginal_x=None,  # type: ignore
                          marginal_y=None)  # type: ignore

        # Speed of crossing vs population of city
        df = df_mapping[df_mapping["speed_crossing"] != 0].copy()
        df = df[(df["traffic_mortality"].notna()) & (df["traffic_mortality"] != 0)]
        df['state'] = df['state'].fillna('NA')
        bivariate.scatter(df=df,
                          x="speed_crossing",
                          y="traffic_mortality",
                          color="continent",
                          text="city",
                          xaxis_title='Mean speed of crossing (in m/s)',
                          yaxis_title='National traffic mortality rate (per 100,000 of population)',
                          pretty_text=False,
                          marker_size=10,
                          save_file=True,
                          hover_data=hover_data,
                          hover_name="city",
                          legend_title="",
                          legend_x=0.87,
                          legend_y=1.0,
                          label_distance_factor=2.0,
                          marginal_x=None,  # type: ignore
                          marginal_y=None)  # type: ignore

        # Time to start crossing vs population of city
        df = df_mapping[df_mapping["time_crossing"] != 0].copy()
        df = df[(df["literacy_rate"].notna()) & (df["literacy_rate"] != 0)]
        df['state'] = df['state'].fillna('NA')
        bivariate.scatter(df=df,
                          x="time_crossing",
                          y="literacy_rate",
                          color="continent",
                          text="city",
                          xaxis_title='Crossing initiation time (in s)',
                          yaxis_title='Literacy rate',
                          pretty_text=False,
                          marker_size=10,
                          save_file=True,
                          hover_data=hover_data,
                          hover_name="city",
                          legend_title="",
                          legend_x=0.87,
                          legend_y=0.01,
                          label_distance_factor=3.0,
                          marginal_x=None,  # type: ignore
                          marginal_y=None)  # type: ignore

        # Speed of crossing vs population of city
        df = df_mapping[df_mapping["speed_crossing"] != 0].copy()
        df = df[(df["literacy_rate"].notna()) & (df["literacy_rate"] != 0)]
        df['state'] = df['state'].fillna('NA')
        bivariate.scatter(df=df,
                          x="speed_crossing",
                          y="literacy_rate",
                          color="continent",
                          text="city",
                          xaxis_title='Mean speed of crossing (in m/s)',
                          yaxis_title='Literacy rate',
                          pretty_text=False,
                          marker_size=10,
                          save_file=True,
                          hover_data=hover_data,
                          hover_name="city",
                          legend_title="",
                          legend_x=0.87,
                          legend_y=0.01,
                          label_distance_factor=3.0,
                          marginal_x=None,  # type: ignore
                          marginal_y=None)  # type: ignore

        # Time to start crossing vs population of city
        df = df_mapping[df_mapping["time_crossing"] != 0].copy()
        df = df[(df["gini"].notna()) & (df["gini"] != 0)]
        df['state'] = df['state'].fillna('NA')
        bivariate.scatter(df=df,
                          x="time_crossing",
                          y="gini",
                          color="continent",
                          text="city",
                          xaxis_title='Crossing initiation time (in s)',
                          yaxis_title='Gini coefficient',
                          pretty_text=False,
                          marker_size=10,
                          save_file=True,
                          hover_data=hover_data,
                          hover_name="city",
                          legend_title="",
                          legend_x=0.87,
                          legend_y=1.0,
                          label_distance_factor=3.0,
                          marginal_x=None,  # type: ignore
                          marginal_y=None)  # type: ignore

        # Speed of crossing vs population of city
        df = df_mapping[df_mapping["speed_crossing"] != 0].copy()
        df = df[(df["gini"].notna()) & (df["gini"] != 0)]
        df['state'] = df['state'].fillna('NA')
        bivariate.scatter(df=df,
                          x="speed_crossing",
                          y="gini",
                          color="continent",
                          text="city",
                          xaxis_title='Mean speed of crossing (in m/s)',
                          yaxis_title='Gini coefficient',
                          pretty_text=False,
                          marker_size=10,
                          save_file=True,
                          hover_data=hover_data,
                          hover_name="city",
                          legend_title="",
                          legend_x=0.87,
                          legend_y=1.0,
                          label_distance_factor=2.0,
                          marginal_x=None,  # type: ignore
                          marginal_y=None)  # type: ignore

        # Time to start crossing vs population of city
        df = df_mapping[df_mapping["time_crossing"] != 0].copy()
        df = df[(df["traffic_index"].notna()) & (df["traffic_index"] != 0)]
        df['state'] = df['state'].fillna('NA')
        bivariate.scatter(df=df,
                          x="time_crossing",
                          y="traffic_index",
                          color="continent",
                          text="city",
                          # size="gmp",
                          xaxis_title='Crossing initiation time (in s)',
                          yaxis_title='Traffic index',
                          pretty_text=False,
                          marker_size=10,
                          save_file=True,
                          hover_data=hover_data,
                          hover_name="city",
                          legend_title="",
                          legend_x=0.87,
                          legend_y=1.0,
                          label_distance_factor=2.0,
                          marginal_x=None,  # type: ignore
                          marginal_y=None)  # type: ignore

        # Speed of crossing vs population of city
        df = df_mapping[df_mapping["speed_crossing"] != 0].copy()
        df = df[df["traffic_index"] != 0]
        df['state'] = df['state'].fillna('NA')
        bivariate.scatter(df=df,
                          x="speed_crossing",
                          y="traffic_index",
                          color="continent",
                          text="city",
                          xaxis_title='Mean speed of crossing (in m/s)',
                          yaxis_title='Traffic index',
                          pretty_text=False,
                          marker_size=10,
                          save_file=True,
                          hover_data=hover_data,
                          hover_name="city",
                          legend_title="",
                          legend_x=0.87,
                          legend_y=1.0,
                          label_distance_factor=2.0,
                          marginal_x=None,  # type: ignore
                          marginal_y=None)  # type: ignore

        # Speed of crossing vs detected mobile phones
        df = df_mapping[df_mapping["time_crossing"] != 0].copy()
        df['state'] = df['state'].fillna('NA')
        df['cellphone_normalised'] = df['cellphone'] / df['total_time']
        bivariate.scatter(df=df,
                          x="time_crossing",
                          y="cellphone_normalised",
                          color="continent",
                          text="city",
                          xaxis_title='Crossing initiation time (in s)',
                          yaxis_title='Mobile phones detected (normalised over time)',
                          pretty_text=False,
                          marker_size=10,
                          save_file=True,
                          hover_data=hover_data,
                          hover_name="city",
                          legend_title="",
                          legend_x=0.87,
                          legend_y=1.0,
                          label_distance_factor=3.0,
                          marginal_x=None,  # type: ignore
                          marginal_y=None)  # type: ignore

        # Speed of crossing vs detected mobile phones
        df = df_mapping[df_mapping["speed_crossing"] != 0].copy()
        df['state'] = df['state'].fillna('NA')
        df['cellphone_normalised'] = df['cellphone'] / df['total_time']
        bivariate.scatter(df=df,
                          x="speed_crossing",
                          y="cellphone_normalised",
                          color="continent",
                          text="city",
                          xaxis_title='Mean speed of crossing (in m/s)',
                          yaxis_title='Mobile phones detected (normalised over time)',
                          pretty_text=False,
                          marker_size=10,
                          save_file=True,
                          hover_data=hover_data,
                          hover_name="city",
                          legend_title="",
                          legend_x=0.87,
                          legend_y=1.0,
                          label_distance_factor=3.0,
                          marginal_x=None,  # type: ignore
                          marginal_y=None)  # type: ignore

        # Jaywalking
        crossing.plot_crossing_without_traffic_light(df_mapping,
                                                     x_axis_title_height=60,
                                                     font_size_captions=common.get_configs("font_size"),
                                                     legend_x=0.97,
                                                     legend_y=1.0,
                                                     legend_spacing=0.004)
        crossing.plot_crossing_with_traffic_light(df_mapping,
                                                  x_axis_title_height=60,
                                                  font_size_captions=common.get_configs("font_size"),
                                                  legend_x=0.97,
                                                  legend_y=1.0,
                                                  legend_spacing=0.004)

        # Crossing with and without traffic lights
        df = df_mapping.copy()
        df['state'] = df['state'].fillna('NA')
        df['with_trf_light_norm'] = (df['with_trf_light_day'] + df['with_trf_light_night']) / df['total_time'] / df['population_city']  # noqa: E501
        df['without_trf_light_norm'] = (df['without_trf_light_day'] + df['without_trf_light_night']) / df['total_time'] / df['population_city']  # noqa: E501
        bivariate.scatter(df=df,
                          x="with_trf_light_norm",
                          y="without_trf_light_norm",
                          color="continent",
                          text="city",
                          xaxis_title='Crossing events with traffic lights (normalised)',
                          yaxis_title='Crossing events without traffic lights (normalised)',
                          pretty_text=False,
                          marker_size=10,
                          save_file=True,
                          hover_data=hover_data,
                          hover_name="city",
                          legend_title="",
                          legend_x=0.87,
                          legend_y=1.0,
                          label_distance_factor=3.0,
                          marginal_x=None,  # type: ignore
                          marginal_y=None)  # type: ignore

    if common.get_configs("analysis_level") == "country":
        df_countries = aggregation.aggregate_by_iso3(df_mapping)
        df_countries_raw = aggregation.aggregate_by_iso3(df_mapping_raw)

        columns_remove = ['videos', 'time_of_day', 'start_time', 'end_time', 'upload_date', 'fps_list', 'vehicle_type']
        hover_data = list(set(df_countries.columns) - set(columns_remove))

        columns_remove_raw = ['gini', 'traffic_mortality', 'avg_height', 'population_country', 'population_city',
                              'med_age', 'literacy_rate']
        hover_data_raw = list(set(df_countries.columns) - set(columns_remove) - set(columns_remove_raw))

        df_countries.to_csv(os.path.join(common.output_dir, "mapping_countries.csv"))

        # Map with images. currently works on a 13" MacBook air screen in chrome, as things are hardcoded...
        maps.map_world(df=df_countries_raw,
                       color="continent",                # same default as map_political
                       show_cities=True,
                       df_cities=df_mapping,
                       show_images=True,
                       hover_data=hover_data_raw,
                       save_file=True,
                       save_final=False,
                       file_name="raw_map")

        # Map with screenshots and countries colours by continent
        maps.map_world(df=df_countries,
                       color="continent",
                       show_cities=True,
                       df_cities=df_mapping,
                       show_images=True,
                       hover_data=hover_data,
                       save_file=False,
                       save_final=False,
                       file_name="map_screenshots",
                       show_colorbar=True,
                       colorbar_title="Continent",
                       colorbar_kwargs=dict(y=0.035, len=0.55, bgcolor="rgba(255,255,255,0.9)"))

        # Map with screenshots and countries colours by amount of footage
        hover_data = list(set(df_countries_raw.columns) - set(columns_remove))

        # log(1 + x) to avoid -inf for zero
        df_countries_raw["log_total_time"] = np.log1p(df_countries_raw["total_time"])

        # Produce map with all data
        df = df_mapping_raw.copy()  # copy df to manipulate for output
        df['state'] = df['state'].fillna('NA')  # Set state to NA

        # Sort by continent and city, both in ascending order
        df = df.sort_values(by=["continent", "city"], ascending=[True, True])

        maps.map_world(df=df_countries_raw,
                       color="log_total_time",
                       show_cities=True,
                       df_cities=df_mapping,             # fixed from df to df_mapping
                       show_images=True,
                       hover_data=hover_data,
                       show_colorbar=True,
                       colorbar_title="Footage (log)",
                       save_file=True,
                       save_final=False,
                       file_name="map_screenshots_total_time")

        df_countries_raw.drop(['speed_crossing_day_country', 'speed_crossing_night_country',
                               'speed_crossing_day_night_country_avg',
                               'time_crossing_day_country', 'time_crossing_night_country',
                               'time_crossing_day_night_country_avg'
                               ], axis=1, inplace=True)
        df_countries_raw.to_csv(os.path.join(common.output_dir, "mapping_countries_raw.csv"))

        # Amount of footage
        bivariate.scatter(df=df_countries,
                          x="total_time",
                          y="person",
                          extension=common.get_configs("analysis_level"),
                          color="continent",
                          text="iso3",
                          xaxis_title='Total time of footage (s)',
                          yaxis_title='Number of detected pedestrians',
                          pretty_text=False,
                          marker_size=10,
                          save_file=True,
                          hover_data=hover_data,
                          hover_name="country",
                          legend_title="",
                          legend_x=0.01,
                          legend_y=1.0,
                          label_distance_factor=0.5,
                          marginal_x=None,  # type: ignore
                          marginal_y=None)  # type: ignore

        # Amount of bicycle footage normalised
        df = df_countries[df_countries["person"] != 0].copy()
        df['person_norm'] = df['person'] / df['total_time']
        bivariate.scatter(df=df,
                          x="total_time",
                          y="person_norm",
                          color="continent",
                          text="iso3",
                          xaxis_title='Total time of footage (s)',
                          yaxis_title='Number of detected pedestrians (normalised over amount of footage)',
                          pretty_text=False,
                          marker_size=10,
                          save_file=True,
                          hover_data=hover_data,
                          hover_name="country",
                          legend_title="",
                          legend_x=0.94,
                          legend_y=1.0,
                          marginal_x=None,  # type: ignore
                          marginal_y=None)  # type: ignore

        # Amount of bicycle footage normalised
        df = df_countries[df_countries["bicycle"] != 0].copy()
        df['bicycle_norm'] = df['bicycle'] / df['total_time']
        bivariate.scatter(df=df,
                          x="total_time",
                          y="bicycle_norm",
                          color="continent",
                          text="iso3",
                          xaxis_title='Total time of footage (s)',
                          yaxis_title='Number of detected bicycle (normalised over amount of footage)',
                          pretty_text=False,
                          marker_size=10,
                          save_file=True,
                          hover_data=hover_data,
                          hover_name="country",
                          legend_title="",
                          legend_x=0.94,
                          legend_y=1.0,
                          marginal_x=None,  # type: ignore
                          marginal_y=None)  # type: ignore

        stacked.stack_plot_country(df_countries,
                                   order_by="average",
                                   metric="time",
                                   data_view="combined",
                                   title_text="Crossing initiation time (s)",
                                   filename="time_crossing_avg_country",
                                   font_size_captions=common.get_configs("font_size") + 8,
                                   legend_x=0.95,
                                   legend_y=0.04,
                                   legend_spacing=0.02,
                                   top_margin=100)

        stacked.stack_plot_country(df_countries,
                                   order_by="condition",
                                   metric="speed",
                                   data_view="combined",
                                   title_text="Mean speed of crossing (in m/s)",
                                   filename="crossing_speed_combined_country",
                                   font_size_captions=common.get_configs("font_size") + 28,
                                   legend_x=0.92,
                                   legend_y=0.04,
                                   legend_spacing=0.02,
                                   top_margin=150,
                                   height=2450,
                                   width=2480)

        stacked.stack_plot_country(df_countries,
                                   order_by="condition",
                                   metric="time",
                                   data_view="combined",
                                   title_text="Mean crossing initiation time (in s)",
                                   filename="time_crossing_combined_country",
                                   font_size_captions=common.get_configs("font_size") + 28,
                                   legend_x=0.92,
                                   legend_y=0.04,
                                   legend_spacing=0.02,
                                   top_margin=150,
                                   height=2400,
                                   width=2480)

        stacked.stack_plot_country(df_countries_raw,
                                   order_by="condition",
                                   metric="speed",
                                   data_view="combined",
                                   title_text="Mean speed of crossing (in m/s)",
                                   filename="crossing_speed_combined_country_raw",
                                   font_size_captions=common.get_configs("font_size") + 28,
                                   raw=True,
                                   legend_x=0.92,
                                   legend_y=0.04,
                                   legend_spacing=0.02,
                                   top_margin=150,
                                   height=2400,
                                   width=2480)

        stacked.stack_plot_country(df_countries_raw,
                                   order_by="condition",
                                   metric="time",
                                   data_view="combined",
                                   title_text="Crossing initiation time (s)",
                                   filename="time_crossing_combined_country_raw",
                                   font_size_captions=common.get_configs("font_size") + 28,
                                   raw=True,
                                   legend_x=0.92,
                                   legend_y=0.04,
                                   legend_spacing=0.02,
                                   top_margin=150,
                                   height=2400,
                                   width=2480)

        stacked.stack_plot_country(df_countries,
                                   order_by="alphabetical",
                                   metric="time",
                                   data_view="combined",
                                   title_text="Crossing initiation time (s)",
                                   filename="time_crossing_alphabetical_country",
                                   font_size_captions=common.get_configs("font_size"),
                                   legend_x=0.94,
                                   legend_y=0.03,
                                   legend_spacing=0.02,
                                   top_margin=100)

        stacked.stack_plot_country(df_countries,
                                   order_by="average",
                                   metric="speed",
                                   data_view="combined",
                                   title_text="Mean speed of crossing (in m/s)",
                                   filename="crossing_speed_avg_country",
                                   font_size_captions=common.get_configs("font_size") + 8,
                                   legend_x=0.87,
                                   legend_y=0.04,
                                   legend_spacing=0.02,
                                   top_margin=100)

        stacked.stack_plot_country(df_countries,
                                   order_by="alphabetical",
                                   metric="speed",
                                   data_view="combined",
                                   title_text="Mean speed of crossing (in m/s)",
                                   filename="crossing_speed_alphabetical_country",
                                   font_size_captions=common.get_configs("font_size"),
                                   legend_x=0.94,
                                   legend_y=0.03,
                                   legend_spacing=0.02,
                                   top_margin=100)

        # Plotting stacked plot during day
        stacked.stack_plot_country(df_countries,
                                   order_by="average",
                                   metric="time",
                                   data_view="day",
                                   title_text="Crossing initiation time (s)",
                                   filename="time_crossing_avg_day_country",
                                   font_size_captions=common.get_configs("font_size"),
                                   top_margin=100)

        stacked.stack_plot_country(df_countries,
                                   order_by="alphabetical",
                                   metric="time",
                                   data_view="day",
                                   title_text="Crossing initiation time (s)",
                                   filename="time_crossing_alphabetical_day_country",
                                   font_size_captions=common.get_configs("font_size"),
                                   top_margin=100)

        stacked.stack_plot_country(df_countries,
                                   order_by="average",
                                   metric="speed",
                                   data_view="day",
                                   title_text="Mean speed of crossing (in m/s)",
                                   filename="crossing_speed_avg_day_country",
                                   font_size_captions=common.get_configs("font_size"),
                                   top_margin=100)

        stacked.stack_plot_country(df_countries,
                                   order_by="alphabetical",
                                   metric="speed",
                                   data_view="day",
                                   title_text="Mean speed of crossing (in m/s)",
                                   filename="crossing_speed_alphabetical_day_country",
                                   font_size_captions=common.get_configs("font_size"),
                                   top_margin=100)

        # Plotting stacked plot during night
        stacked.stack_plot_country(df_countries,
                                   order_by="average",
                                   metric="time",
                                   data_view="night",
                                   title_text="Crossing initiation time (s)",
                                   filename="time_crossing_avg_night_country",
                                   font_size_captions=common.get_configs("font_size"))

        stacked.stack_plot_country(df_countries,
                                   order_by="alphabetical",
                                   metric="time",
                                   data_view="night",
                                   title_text="Crossing initiation time (s)",
                                   filename="time_crossing_alphabetical_night_country",
                                   font_size_captions=common.get_configs("font_size"),
                                   top_margin=100)

        stacked.stack_plot_country(df_countries,
                                   order_by="average",
                                   metric="speed",
                                   data_view="night",
                                   title_text="Mean speed of crossing (in m/s)",
                                   filename="crossing_speed_avg_night_country",
                                   font_size_captions=common.get_configs("font_size"),
                                   top_margin=100)

        stacked.stack_plot_country(df_countries,
                                   order_by="alphabetical",
                                   metric="speed",
                                   data_view="night",
                                   title_text="Mean speed of crossing (in m/s)",
                                   filename="crossing_speed_alphabetical_night_country",
                                   font_size_captions=common.get_configs("font_size"),
                                   top_margin=100)

        crossing.speed_and_time_to_start_cross_country(df_countries,
                                                       x_axis_title_height=110,
                                                       font_size_captions=common.get_configs("font_size") + 8,
                                                       legend_x=0.87,
                                                       legend_y=0.04,
                                                       legend_spacing=0.01)

        correlation.correlation_matrix_country(df_mapping, df_countries, pedestrian_cross_city, person_city,
                                               bicycle_city, car_city, motorcycle_city, bus_city, truck_city,
                                               cross_evnt_city, vehicle_city, cellphone_city, traffic_sign_city,
                                               avg_speed_country, avg_time_country,
                                               crossings_without_traffic_equipment_country)

        # Speed of crossing vs Crossing initiation time
        df = df_countries[df_countries["speed_crossing_day_night_country_avg"] != 0].copy()
        df = df[df["time_crossing_day_night_country_avg"] != 0]
        bivariate.scatter(df=df,
                          x="speed_crossing_day_night_country_avg",
                          y="time_crossing_day_night_country_avg",
                          color="continent",
                          text="iso3",
                          xaxis_title='Mean speed of crossing (in m/s)',
                          yaxis_title='Crossing initiation time (s)',
                          pretty_text=False,
                          marker_size=10,
                          save_file=True,
                          hover_data=hover_data,
                          hover_name="country",
                          legend_title="",
                          legend_x=0.87,
                          legend_y=1.0,
                          label_distance_factor=0.5,
                          marginal_x=None,  # type: ignore
                          marginal_y=None)  # type: ignore

        # Speed of crossing during daytime vs time to start crossing during daytime
        df = df_countries[df_countries["speed_crossing_day_country"] != 0].copy()
        df = df[df["time_crossing_day_country"] != 0]
        bivariate.scatter(df=df,
                          x="speed_crossing_day_country",
                          y="time_crossing_day_country",
                          color="continent",
                          text="iso3",
                          xaxis_title='Crossing speed during daytime (in m/s)',
                          yaxis_title='Crossing initiation time during daytime (in s)',
                          pretty_text=False,
                          marker_size=10,
                          save_file=True,
                          hover_data=hover_data,
                          hover_name="country",
                          legend_title="",
                          legend_x=0.87,
                          legend_y=1.0,
                          label_distance_factor=0.5,
                          marginal_x=None,  # type: ignore
                          marginal_y=None)  # type: ignore

        # Speed of crossing during night time vs time to start crossing during night time
        df = df_countries[df_countries["speed_crossing_night_country"] != 0].copy()
        df = df[df["time_crossing_night_country"] != 0]
        bivariate.scatter(df=df,
                          x="speed_crossing_night_country",
                          y="time_crossing_night_country",
                          color="continent",
                          text="iso3",
                          xaxis_title='Crossing speed during night time (in m/s)',
                          yaxis_title='Crossing initiation time during night time (in s)',
                          pretty_text=False,
                          marker_size=10,
                          save_file=True,
                          hover_data=hover_data,
                          hover_name="country",
                          legend_title="",
                          legend_x=0.87,
                          legend_y=1.0,
                          label_distance_factor=0.5,
                          marginal_x=None,  # type: ignore
                          marginal_y=None)  # type: ignore

        # Time to start crossing vs population of city
        df = df_countries[df_countries["time_crossing_day_night_country_avg"] != 0].copy()
        df = df[(df["population_country"].notna()) & (df["population_country"] != 0)]
        bivariate.scatter(df=df,
                          x="time_crossing_day_night_country_avg",
                          y="population_country",
                          color="continent",
                          text="iso3",
                          xaxis_title='Crossing initiation time (s)',
                          yaxis_title='Population of country',
                          pretty_text=False,
                          marker_size=10,
                          save_file=True,
                          hover_data=hover_data,
                          hover_name="country",
                          legend_title="",
                          legend_x=0.87,
                          legend_y=1.0,
                          label_distance_factor=0.5,
                          marginal_x=None,  # type: ignore
                          marginal_y=None)  # type: ignore

        # Speed of crossing vs population of country
        df = df_countries[df_countries["speed_crossing_day_night_country_avg"] != 0].copy()
        df = df[(df["population_country"].notna()) & (df["population_country"] != 0)]
        bivariate.scatter(df=df,
                          x="speed_crossing_day_night_country_avg",
                          y="population_country",
                          color="continent",
                          text="iso3",
                          xaxis_title='Mean speed of crossing (in m/s)',
                          yaxis_title='Population of country',
                          pretty_text=False,
                          marker_size=10,
                          save_file=True,
                          hover_data=hover_data,
                          hover_name="country",
                          legend_title="",
                          legend_x=0.87,
                          legend_y=1.0,
                          label_distance_factor=0.2,
                          marginal_x=None,  # type: ignore
                          marginal_y=None)  # type: ignore

        # Time to start crossing vs population of city
        df = df_countries[df_countries["time_crossing_day_night_country_avg"] != 0].copy()
        df = df[(df["traffic_mortality"].notna()) & (df["traffic_mortality"] != 0)]
        bivariate.scatter(df=df,
                          x="time_crossing_day_night_country_avg",
                          y="traffic_mortality",
                          color="continent",
                          text="iso3",
                          xaxis_title='Crossing initiation time (in s)',
                          yaxis_title='National traffic mortality rate (per 100,000 of population)',
                          pretty_text=False,
                          marker_size=10,
                          save_file=True,
                          hover_data=hover_data,
                          hover_name="country",
                          legend_title="",
                          legend_x=0.87,
                          legend_y=1.0,
                          label_distance_factor=0.5,
                          marginal_x=None,  # type: ignore
                          marginal_y=None)  # type: ignore

        # Speed of crossing vs population of city
        df = df_countries[df_countries["speed_crossing_day_night_country_avg"] != 0].copy()
        df = df[(df["traffic_mortality"].notna()) & (df["traffic_mortality"] != 0)]
        bivariate.scatter(df=df,
                          x="speed_crossing_day_night_country_avg",
                          y="traffic_mortality",
                          color="continent",
                          text="iso3",
                          xaxis_title='Mean speed of crossing (in m/s)',
                          yaxis_title='National traffic mortality rate (per 100,000 of population)',
                          pretty_text=False,
                          marker_size=10,
                          save_file=True,
                          hover_data=hover_data,
                          hover_name="country",
                          legend_title="",
                          legend_x=0.87,
                          legend_y=1.0,
                          label_distance_factor=0.3,
                          marginal_x=None,  # type: ignore
                          marginal_y=None)  # type: ignore

        # Time to start crossing vs population of city
        df = df_countries[df_countries["time_crossing_day_night_country_avg"] != 0].copy()
        df = df[(df["literacy_rate"].notna()) & (df["literacy_rate"] != 0)]
        bivariate.scatter(df=df,
                          x="time_crossing_day_night_country_avg",
                          y="literacy_rate",
                          color="continent",
                          text="iso3",
                          xaxis_title='Crossing initiation time (in s)',
                          yaxis_title='Literacy rate',
                          pretty_text=False,
                          marker_size=10,
                          save_file=True,
                          hover_data=hover_data,
                          hover_name="country",
                          legend_title="",
                          legend_x=0.87,
                          legend_y=0.01,
                          label_distance_factor=0.5,
                          marginal_x=None,  # type: ignore
                          marginal_y=None)  # type: ignore

        # Speed of crossing vs population of city
        df = df_countries[df_countries["speed_crossing_day_night_country_avg"] != 0].copy()
        df = df[(df["literacy_rate"].notna()) & (df["literacy_rate"] != 0)]
        bivariate.scatter(df=df,
                          x="speed_crossing_day_night_country_avg",
                          y="literacy_rate",
                          color="continent",
                          text="iso3",
                          xaxis_title='Mean speed of crossing (in m/s)',
                          yaxis_title='Literacy rate',
                          pretty_text=False,
                          marker_size=10,
                          save_file=True,
                          hover_data=hover_data,
                          hover_name="country",
                          legend_title="",
                          legend_x=0.87,
                          legend_y=0.01,
                          label_distance_factor=0.4,
                          marginal_x=None,  # type: ignore
                          marginal_y=None)  # type: ignore

        # Time to start crossing vs population of city
        df = df_countries[df_countries["time_crossing_day_night_country_avg"] != 0].copy()
        df = df[(df["gini"].notna()) & (df["gini"] != 0)]
        bivariate.scatter(df=df,
                          x="time_crossing_day_night_country_avg",
                          y="gini",
                          color="continent",
                          text="iso3",
                          xaxis_title='Crossing initiation time (in s)',
                          yaxis_title='Gini coefficient',
                          pretty_text=False,
                          marker_size=10,
                          save_file=True,
                          hover_data=hover_data,
                          hover_name="country",
                          legend_title="",
                          legend_x=0.87,
                          legend_y=1.0,
                          label_distance_factor=0.5,
                          marginal_x=None,  # type: ignore
                          marginal_y=None)  # type: ignore

        # Speed of crossing vs population of city
        df = df_countries[df_countries["speed_crossing_day_night_country_avg"] != 0].copy()
        df = df[(df["gini"].notna()) & (df["gini"] != 0)]
        bivariate.scatter(df=df,
                          x="speed_crossing_day_night_country_avg",
                          y="gini",
                          color="continent",
                          text="iso3",
                          xaxis_title='Mean speed of crossing (in m/s)',
                          yaxis_title='Gini coefficient',
                          pretty_text=False,
                          marker_size=10,
                          save_file=True,
                          hover_data=hover_data,
                          hover_name="country",
                          legend_title="",
                          legend_x=0.87,
                          legend_y=1.0,
                          label_distance_factor=0.5,
                          marginal_x=None,  # type: ignore
                          marginal_y=None)  # type: ignore

        # Time to start crossing vs population of city
        df = df_countries[df_countries["time_crossing_day_night_country_avg"] != 0].copy()
        df = df[(df["med_age"].notna()) & (df["med_age"] != 0)]
        bivariate.scatter(df=df,
                          x="time_crossing_day_night_country_avg",
                          y="med_age",
                          color="continent",
                          text="iso3",
                          # size="gmp",
                          xaxis_title='Crossing initiation time (in s)',
                          yaxis_title='Median age (in years)',
                          pretty_text=False,
                          marker_size=10,
                          save_file=True,
                          hover_data=hover_data,
                          hover_name="country",
                          legend_title="",
                          legend_x=0.87,
                          legend_y=1.0,
                          label_distance_factor=0.5,
                          marginal_x=None,  # type: ignore
                          marginal_y=None)  # type: ignore

        # Speed of crossing vs population of city
        df = df_countries[df_countries["speed_crossing_day_night_country_avg"] != 0].copy()
        df = df[df["med_age"] != 0]
        bivariate.scatter(df=df,
                          x="speed_crossing_day_night_country_avg",
                          y="med_age",
                          color="continent",
                          text="iso3",
                          xaxis_title='Mean speed of crossing (in m/s)',
                          yaxis_title='Median age (in years)',
                          pretty_text=False,
                          marker_size=10,
                          save_file=True,
                          hover_data=hover_data,
                          hover_name="country",
                          legend_title="",
                          legend_x=0.87,
                          legend_y=1.0,
                          label_distance_factor=0.4,
                          marginal_x=None,  # type: ignore
                          marginal_y=None)  # type: ignore

        # Speed of crossing vs detected mobile phones
        df = df_countries[df_countries["time_crossing_day_night_country_avg"] != 0].copy()
        df['cellphone_normalised'] = df['cellphone'] / df['total_time']
        bivariate.scatter(df=df,
                          x="time_crossing_day_night_country_avg",
                          y="cellphone_normalised",
                          color="continent",
                          text="iso3",
                          xaxis_title='Crossing initiation time (in s)',
                          yaxis_title='Mobile phones detected (normalised over time)',
                          pretty_text=False,
                          marker_size=10,
                          save_file=True,
                          hover_data=hover_data,
                          hover_name="country",
                          legend_title="",
                          legend_x=0.87,
                          legend_y=1.0,
                          label_distance_factor=0.5,
                          marginal_x=None,  # type: ignore
                          marginal_y=None)  # type: ignore

        # Speed of crossing vs detected mobile phones
        df = df_countries[df_countries["speed_crossing_day_night_country_avg"] != 0].copy()
        df['cellphone_normalised'] = df['cellphone'] / df['total_time']
        bivariate.scatter(df=df,
                          x="speed_crossing_day_night_country_avg",
                          y="cellphone_normalised",
                          color="continent",
                          text="iso3",
                          xaxis_title='Mean speed of crossing (in m/s)',
                          yaxis_title='Mobile phones detected (normalised over time)',
                          pretty_text=False,
                          marker_size=10,
                          save_file=True,
                          hover_data=hover_data,
                          hover_name="country",
                          legend_title="",
                          legend_x=0.87,
                          legend_y=1.0,
                          label_distance_factor=0.5,
                          marginal_x=None,  # type: ignore
                          marginal_y=None)  # type: ignore

        # Mean speed of crossing (used to be plots_class.map)
        maps.map_world(df=df_countries,
                       color="speed_crossing_day_night_country_avg",
                       title="Mean speed of crossing (in m/s)",
                       show_colorbar=True,
                       colorbar_title="",                 # keep your empty title behavior
                       filter_zero_nan=True,              # preserves old map() filtering
                       save_file=True,
                       file_name="map_speed_crossing"
                       )

        # Crossing initiation time (used to be plots_class.map)
        maps.map_world(df=df_countries,
                       color="time_crossing_day_night_country_avg",
                       title="Crossing initiation time (in s)",
                       show_colorbar=True,
                       colorbar_title="",
                       filter_zero_nan=True,
                       save_file=True,
                       file_name="map_crossing_time"
                       )

        # Crossing with and without traffic lights
        df = df_countries.copy()
        # df['state'] = df['state'].fillna('NA')
        df['with_trf_light_norm'] = (df['with_trf_light_day_country'] + df['with_trf_light_night_country']) / df['total_time'] / df['population_country']  # noqa: E501
        df['without_trf_light_norm'] = (df['without_trf_light_day_country'] + df['without_trf_light_night_country']) / df['total_time'] / df['population_country']  # noqa: E501
        df['country'] = df['country'].str.title()
        bivariate.scatter(df=df,
                          x="with_trf_light_norm",
                          y="without_trf_light_norm",
                          color="continent",
                          text="iso3",
                          xaxis_title='Crossing events with traffic lights (normalised)',
                          yaxis_title='Crossing events without traffic lights (normalised)',
                          pretty_text=False,
                          marker_size=10,
                          save_file=True,
                          hover_data=hover_data,
                          hover_name="country",
                          legend_title="",
                          legend_x=0.87,
                          legend_y=1.0,
                          label_distance_factor=0.5,
                          marginal_x=None,  # type: ignore
                          marginal_y=None)  # type: ignore

        # Exclude zero values before finding min
        nonzero_speed = df_countries[df_countries["speed_crossing_day_night_country_avg"] > 0]
        nonzero_time = df_countries[df_countries["time_crossing_day_night_country_avg"] > 0]

        max_speed_idx = df_countries["speed_crossing_day_night_country_avg"].idxmax()
        min_speed_idx = nonzero_speed["speed_crossing_day_night_country_avg"].idxmin()

        max_time_idx = df_countries["time_crossing_day_night_country_avg"].idxmax()
        min_time_idx = nonzero_time["time_crossing_day_night_country_avg"].idxmin()

        # Mean and standard deviation
        speed_mean = nonzero_speed["speed_crossing_day_night_country_avg"].mean()
        speed_std = nonzero_speed["speed_crossing_day_night_country_avg"].std()

        time_mean = nonzero_time["time_crossing_day_night_country_avg"].mean()
        time_std = nonzero_time["time_crossing_day_night_country_avg"].std()

        logger.info(f"Country with the highest average speed while crossing: {df_countries.loc[max_speed_idx, 'country']} "  # noqa:E501
                    f"({df_countries.loc[max_speed_idx, 'speed_crossing_day_night_country_avg']:.2f})")

        logger.info(f"Country with the lowest non-zero average speed while crossing: {nonzero_speed.loc[min_speed_idx, 'country']} "  # noqa:E501
                    f"({nonzero_speed.loc[min_speed_idx, 'speed_crossing_day_night_country_avg']:.2f})")

        logger.info(f"Mean speed while crossing (non-zero): {speed_mean:.2f}")
        logger.info(f"Standard deviation of speed while crossing (non-zero): {speed_std:.2f}")

        logger.info(f"Country with the highest average crossing time: {df_countries.loc[max_time_idx, 'country']} "
                    f"({df_countries.loc[max_time_idx, 'time_crossing_day_night_country_avg']:.2f})")

        logger.info(f"Country with the lowest non-zero average crossing time: {nonzero_time.loc[min_time_idx, 'country']} "  # noqa: E501
                    f"({nonzero_time.loc[min_time_idx, 'time_crossing_day_night_country_avg']:.2f})")

        logger.info(f"Mean crossing time (non-zero): {time_mean:.2f}")
        logger.info(f"Standard deviation of crossing time (non-zero): {time_std:.2f}")

        stats = df_countries[['total_time', 'total_videos']].agg(['mean', 'std', 'sum'])

        logger.info(
            f"Average total_time: {stats.loc['mean', 'total_time']:.2f}, "
            f"Standard deviation: {stats.loc['std', 'total_time']:.2f}, "
            f"Sum: {stats.loc['sum', 'total_time']:.2f}"
        )
        logger.info(
            f"Average total_videos: {stats.loc['mean', 'total_videos']:.2f}, "
            f"Standard deviation: {stats.loc['std', 'total_videos']:.2f}, "
            f"Sum: {stats.loc['sum', 'total_videos']:.2f}"
        )

        # Max total_time
        max_row = df_countries.loc[df_countries['total_time'].idxmax()]
        logger.info(
            f"Country with maximum total_time: {max_row['country']}, "
            f"total_time: {max_row['total_time']}, "
            f"total_videos: {max_row['total_videos']}"
        )

        # Min total_time
        min_row = df_countries.loc[df_countries['total_time'].idxmin()]
        logger.info(
            f"Country with minimum total_time: {min_row['country']}, "
            f"total_time: {min_row['total_time']}, "
            f"total_videos: {min_row['total_videos']}"
        )
