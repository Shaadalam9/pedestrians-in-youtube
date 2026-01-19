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
import warnings
from typing import Set, Optional, Dict

import polars as pl
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
        df_mapping = pl.read_csv(common.get_configs("mapping"))

        # Produce map with all data
        df = df_mapping.clone()  # copy df to manipulate for output
        df = df.with_columns(pl.col("state").fill_null("NA").alias("state"))

        # Sort by continent and city, both in ascending order
        df = df.sort(by=["continent", "city"])

        # Count of videos (handles: [id], "[id1,id2]", and [] -> 0)
        videos_clean = (
            pl.col("videos")
              .cast(pl.Utf8)
              .str.strip_chars("\"'")   # remove surrounding quotes if present
              .str.strip_chars("[]")     # remove surrounding brackets
              .str.strip_chars()         # trim whitespace
        )

        df = df.with_columns(
            pl.when(pl.col("videos").is_null() | (videos_clean == ""))
              .then(0)
              .otherwise(
                  videos_clean
                  .str.split(",")
                  .list.eval(pl.element().str.strip_chars())  # trim each item
                  .list.filter(pl.element() != "")            # drop empties (so [] -> 0)
                  .list.len()
              ).alias("video_count")
        )

        # Total amount of seconds in segments
        def flatten(lst):
            """Flattens nested lists like [[1, 2], [3, 4]] -> [1, 2, 3, 4]"""
            out = []
            for sub in lst:
                if isinstance(sub, list):
                    out.extend(sub)
                else:
                    out.append(sub)
            return out

        def compute_total_time(row: dict) -> int:
            try:
                start_raw = row.get("start_time")
                end_raw = row.get("end_time")

                start_times = flatten(ast.literal_eval(start_raw)) if start_raw is not None else []
                end_times = flatten(ast.literal_eval(end_raw)) if end_raw is not None else []

                return int(sum(e - s for s, e in zip(start_times, end_times)))
            except Exception as e:
                logger.error(f"Error in row {row.get('id')}: {e}")
                return 0

        df = df.with_columns(
            pl.struct(["id", "start_time", "end_time"])
              .map_elements(compute_total_time, return_dtype=pl.Int64)
              .alias("total_time")
        )

        # create flag_city column
        flag_expr = pl.col("iso3").map_elements(
            lambda x: analysis_class.iso3_to_flag.get(x, "🏳️"),
            return_dtype=pl.Utf8,
        )

        # Create a new country label with emoji flag + country name
        df = df.with_columns([
            pl.concat_str([flag_expr, pl.col("city").cast(pl.Utf8)], separator=" ").alias("flag_city"),
            pl.concat_str([flag_expr, pl.col("country").cast(pl.Utf8)], separator=" ").alias("flag_country"),
        ])

        # Data to avoid showing on hover in scatter plots
        columns_remove = ['videos', 'time_of_day', 'start_time', 'end_time', 'upload_date', 'vehicle_type', 'channel',
                          'display_label', 'flag_city', 'flag_country']

        hover_data = sorted(list(set(df.columns) - set(columns_remove)))

        # Sort by continent and city, both in ascending order
        df = df.sort(["continent", "country"])

        # map with all cities
        maps.mapbox_map(df=df.to_pandas(), hover_data=hover_data, hover_name="flag_city", file_name='mapbox_map_all')

        # Sort by continent and city, both in ascending order
        df = df.sort(["country", "city"])

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
        videos_raw = pl.col("videos").cast(pl.Utf8).str.strip_chars()
        videos_unquoted = videos_raw.str.strip_chars("\"'")

        videos_is_list = videos_unquoted.str.starts_with("[") & videos_unquoted.str.ends_with("]")

        videos_inner = (
            videos_unquoted
            .str.strip_chars("[]")
            .str.replace_all(r"[\"']", "")   # remove any inner quotes
            .str.replace_all(r"\s+", "")     # remove whitespace
            .str.strip_chars()
        )

        city_video_count_expr = (
            pl.when(videos_is_list & (videos_inner != ""))
              .then(
                  videos_inner
                  .str.split(",")
                  .list.filter(pl.element() != "")
                  .list.len()
              ).otherwise(0).cast(pl.Int64).alias("city_video_count")
        )

        # Normalize end_time string and sum all numbers found (nested-safe)
        end_raw = pl.col("end_time").cast(pl.Utf8).str.strip_chars()
        end_unquoted = end_raw.str.strip_chars("\"'")
        end_is_list = end_unquoted.str.starts_with("[")

        city_total_time_expr = (
            pl.when(end_is_list)
              .then(
                  end_unquoted
                  .str.extract_all(r"\d+")
                  .list.eval(pl.element().cast(pl.Int64))
                  .list.sum()
                  .fill_null(0)
              ).otherwise(0).cast(pl.Int64).alias("city_total_time")
        )

        df = df.with_columns([city_video_count_expr, city_total_time_expr])

        # ---------- Aggregate to country level ----------
        df_country = (
            df.group_by(["country", "iso3", "continent"])
              .agg([
                  pl.col("city_total_time").sum().alias("total_time"),
                  pl.col("city_video_count").sum().alias("video_count"),
              ])
        )

        # add flag + iso3 label
        df_country = df_country.with_columns(
            pl.concat_str(
                [
                    pl.col("iso3").map_elements(
                        lambda x: analysis_class.iso3_to_flag.get(x, "🏳️"),
                        return_dtype=pl.Utf8,
                    ),
                    pl.col("iso3").cast(pl.Utf8),
                ],
                separator=" ",
            ).alias("flag_country")
        )

        # sort for readability
        df_country = df_country.sort(["continent", "country"])

        # define hover data
        hover_data = ["country", "continent", "total_time", "video_count"]

        # plot (convert at plotting boundary)
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
        distribution.video_histogram_by_month(df=df.to_pandas(),
                                              video_count_col='video_count',
                                              upload_date_col='upload_date',
                                              xaxis_title='Upload month (year-month)',
                                              yaxis_title='Number of videos',
                                              save_file=True)

        # maps with all cities and population heatmap
        maps.mapbox_map(df=df.to_pandas(),
                        hover_data=hover_data,
                        density_col='population_city',
                        density_radius=10,
                        file_name='mapbox_map_all_pop')

        # maps with all cities and video count heatmap
        maps.mapbox_map(df=df.to_pandas(),
                        hover_data=hover_data,
                        density_col='video_count',
                        density_radius=10,
                        file_name='mapbox_map_all_videos')

        # maps with all cities and total time heatmap
        maps.mapbox_map(df=df.to_pandas(),
                        hover_data=hover_data,
                        density_col='total_time',
                        density_radius=10,
                        file_name='mapbox_map_all_time')

        # Type of vehicle over time of day
        df = df_mapping.clone()  # copy df to manipulate for output

        # --- expand rows so each video becomes one row ---
        # Return type: List[Struct{vehicle_type: Utf8, time_of_day: Utf8}]
        pair_dtype = pl.List(
            pl.Struct([
                pl.Field("vehicle_type", pl.Utf8),
                pl.Field("time_of_day", pl.Utf8),
            ])
        )

        def expand_pairs(vs: str | None, ts: str | None) -> list[dict]:
            """Parse stringified lists (possibly nested) and emit expanded (vehicle_type,
               time_of_day) pairs as strings."""
            try:
                vehicle_types = ast.literal_eval(vs) if isinstance(vs, str) else None
                times_of_day = ast.literal_eval(ts) if isinstance(ts, str) else None
                if not (isinstance(vehicle_types, list) and isinstance(times_of_day, list)):
                    return []
            except Exception:
                return []

            out: list[dict] = []
            for v_type, tod in zip(vehicle_types, times_of_day):
                v_list = v_type if isinstance(v_type, list) else [v_type]
                t_list = tod if isinstance(tod, list) else [tod]
                for vt in v_list:
                    for t in t_list:
                        out.append({"vehicle_type": str(vt), "time_of_day": str(t)})
            return out

        def map_with_fallback(dct: dict, v):
            """
            Robust dict lookup for values that may arrive as str/int/float (or numeric strings).
            Tries:
              1) direct key
              2) string key (stripped)
              3) int key (from int(v) or int(float(v)) for "1.0")
              4) float key (rare, but safe)
            Returns None if no match.
            """
            if v is None:
                return None

            # 1) direct key
            if v in dct:
                return dct[v]

            # Normalize string form
            sv = v.strip() if isinstance(v, str) else str(v).strip()

            # 2) string key
            if sv in dct:
                return dct[sv]

            # 3) int key (handle "1" and "1.0")
            try:
                iv = int(sv)
                if iv in dct:
                    return dct[iv]
            except Exception:
                try:
                    iv = int(float(sv))
                    if iv in dct:
                        return dct[iv]
                except Exception:
                    pass

            # 4) float key (less common, but harmless)
            try:
                fv = float(sv)
                if fv in dct:
                    return dct[fv]
            except Exception:
                pass

            return None

        # --- expand rows ---
        df_expanded = (
            df.select(["vehicle_type", "time_of_day"])
              .with_columns(
                  pl.struct(["vehicle_type", "time_of_day"])
                  .map_elements(
                        lambda r: expand_pairs(r["vehicle_type"], r["time_of_day"]),
                        return_dtype=pair_dtype,
                    ).alias("pairs")
              ).select("pairs")               # avoid duplicate column name collisions
               .explode("pairs")
               .with_columns([
                  pl.col("pairs").struct.field("vehicle_type").alias("vehicle_type"),
                  pl.col("pairs").struct.field("time_of_day").alias("time_of_day"),
                  ]).drop("pairs")
        )

        # --- map to human-readable labels ---
        df_expanded = df_expanded.with_columns([
            pl.col("vehicle_type").map_elements(
                lambda x: map_with_fallback(analysis_class.vehicle_map, x),
                return_dtype=pl.Utf8,
            ).alias("vehicle_type_name"),
            pl.col("time_of_day").map_elements(
                lambda x: map_with_fallback(analysis_class.time_map, x),
                return_dtype=pl.Utf8,
            ).alias("time_of_day_name"),
        ])

        # drop rows where mapping failed
        df_expanded = df_expanded.filter(
            pl.col("vehicle_type_name").is_not_null() & pl.col("time_of_day_name").is_not_null()
        )

        # --- aggregate counts ---
        df_summary = (
            df_expanded
            .group_by(["vehicle_type_name", "time_of_day_name"])
            .len()
            .rename({"len": "count"})
        )

        # --- pivot into wide format for stacked bar plot ---
        df_pivot = df_summary.pivot(
            index="vehicle_type_name",
            on="time_of_day_name",      # renamed from `columns`
            values="count",
            aggregate_function="first",
        ).fill_null(0)

        # ensure consistent order of vehicle types
        vehicle_order = [
            "Car", "Bus", "Truck", "Two-wheeler", "Bicycle", "Automated car", "Automated bus", "Automated truck",
            "Automated two-wheeler", "Electric scooter"
        ]
        order_map = {name: i for i, name in enumerate(vehicle_order)}

        df_pivot = (
            df_pivot
            .with_columns(
                pl.col("vehicle_type_name")
                  .map_elements(lambda x: order_map.get(x, 10**9), return_dtype=pl.Int64)
                  .alias("_order")
            )
            .sort("_order")
            .drop("_order")
        )
        # --- plot ---
        distribution.bar(
            df=df_pivot.to_pandas(),
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
        df = df_mapping.clone()  # copy df to manipulate for output

        # --- expand rows so each video becomes one row ---
        pair_dtype = pl.List(
            pl.Struct([
                pl.Field("continent", pl.Utf8),
                pl.Field("time_of_day", pl.Utf8),
            ])
        )

        def expand_continent_tod(continent: str | None, ts: str | None) -> list[dict]:
            try:
                times_of_day = ast.literal_eval(ts) if isinstance(ts, str) else None
                if not isinstance(times_of_day, list):
                    return []
            except Exception:
                return []

            cont = "" if continent is None else str(continent)

            out: list[dict] = []
            for tod in times_of_day:
                t_list = tod if isinstance(tod, list) else [tod]
                for t in t_list:
                    out.append({"continent": cont, "time_of_day": str(t)})
            return out

        # --- expand rows so each time-of-day entry becomes one row ---
        df_expanded = (
            df.select(["continent", "time_of_day"])
              .with_columns(
                  pl.struct(["continent", "time_of_day"])
                  .map_elements(
                        lambda r: expand_continent_tod(r["continent"], r["time_of_day"]),
                        return_dtype=pair_dtype,
                    ).alias("pairs")
              ).select("pairs").explode("pairs").with_columns([
                  pl.col("pairs").struct.field("continent").alias("continent"),
                  pl.col("pairs").struct.field("time_of_day").alias("time_of_day"),
              ]).drop("pairs")
        )

        # --- map to human-readable labels ---
        df_expanded = df_expanded.with_columns(
            pl.col("time_of_day").map_elements(
                lambda x: map_with_fallback(analysis_class.time_map, x),
                return_dtype=pl.Utf8,
            ).alias("time_of_day_name")
        )

        # drop rows where mapping failed
        df_expanded = df_expanded.filter(
            pl.col("time_of_day_name").is_not_null() & pl.col("continent").is_not_null() & (pl.col("continent") != "")
        )

        # --- aggregate counts ---
        df_summary = (
            df_expanded
            .group_by(["continent", "time_of_day_name"])
            .len()
            .rename({"len": "count"})
        )

        # --- pivot into wide format for stacked bar plot ---
        df_pivot = (
            df_summary
            .pivot(
                index="continent",
                on="time_of_day_name",   # Polars >= 1.0.0 uses `on` (not `columns`)
                values="count",
                aggregate_function="first",
            )
            .fill_null(0)
        )

        # ensure only expected columns (and ensure they exist)
        for col in ["Day", "Night"]:
            if col not in df_pivot.columns:
                df_pivot = df_pivot.with_columns(pl.lit(0).alias(col))

        # --- enforce alphabetical continent order ---
        df_pivot = df_pivot.sort("continent")

        time_columns = [col for col in ["Day", "Night"] if col in df_pivot.columns]

        # --- plot ---
        distribution.bar(
            df=df_pivot.to_pandas(),
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

        country, number, _ = metrics_cache.get_unique_values(df_mapping, "iso3")
        logger.info(f"Total number of countries and territories before filtering: {number}.")

        city_state_iso3, number, dup_report = metrics_cache.get_unique_values(
            df_mapping,
            ["city", "state", "iso3"],
            return_duplicates=True,
        )

        logger.info(f"Total number of unique city+state+iso3 keys: {number}.")

        if dup_report is not None and dup_report.height > 0:
            logger.warning(f"Duplicated keys:\n{dup_report}")

        # Limit countries if required
        countries_include = common.get_configs("countries_analyse")
        if countries_include:
            df_mapping = df_mapping.filter(pl.col("iso3").is_in(countries_include))

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
        df_mapping = df_mapping.with_columns(
            [pl.lit(v).alias(k) for k, v in city_country_cols.items()]
        )

        # Precompute fast lookup once (place this immediately before the loops)
        id_to_place: dict[int, tuple[str, str, str]] = {
            int(row_id): (city, state, country)
            for row_id, city, state, country in df_mapping.select(["id", "city", "state", "country"]).iter_rows()
        }

        all_speed = {}
        all_time = {}

        logger.info("Processing csv files.")
        pedestrian_crossing_count, data = {}, {}
        pedestrian_crossing_count_all = {}

        min_conf = common.get_configs("min_confidence")

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

                    file_str: str = os.fspath(filtered)

                    if file_str in MISC_FILES:
                        continue

                    filename_no_ext = os.path.splitext(file_str)[0]
                    logger.debug(f"{filename_no_ext}: fetching values.")

                    file_path = os.path.join(subfolder_path, file_str)

                    # Polars read + filter
                    df = pl.read_csv(file_path)
                    df = df.filter(pl.col("confidence") >= min_conf)

                    # After reading the file, clean up the filename
                    base_name = tools.clean_csv_filename(file_str)
                    filename_no_ext = os.path.splitext(base_name)[0]

                    try:
                        video_id, start_index, fps = filename_no_ext.rsplit("_", 2)
                    except ValueError:
                        logger.warning(f"Unexpected filename format: {filename_no_ext}")
                        continue

                    video_city_id = geo.find_city_id(df_mapping, video_id, int(start_index))

                    place = id_to_place.get(int(video_city_id)) if video_city_id is not None else None
                    if place is None:
                        logger.warning(f"{file_str}: no mapping row found for id={video_city_id}.")
                        continue

                    video_city, video_state, video_country = place
                    logger.debug(f"{file_str}: found values {video_city}, {video_state}, {video_country}.")

                    # Get the number of number and unique id of the object crossing the road
                    # ids give the unique of the person who cross the road after applying the filter, while
                    # all_ids gives every unique_id of the person who crosses the road
                    ids, all_ids = detection.pedestrian_crossing(
                        df,
                        filename_no_ext,
                        df_mapping,
                        common.get_configs("boundary_left"),
                        common.get_configs("boundary_right"),
                        person_id=0,
                    )

                    # Saving it in a dictionary in: {video-id_time: count, ids}
                    pedestrian_crossing_count[filename_no_ext] = {"ids": ids}
                    pedestrian_crossing_count_all[filename_no_ext] = {"ids": all_ids}

                    # Saves the time to cross in form {name_time: {id(s): time(s)}}
                    temp_data = metrics.time_to_cross(
                        df,
                        pedestrian_crossing_count[filename_no_ext]["ids"],
                        filename_no_ext,
                        df_mapping,
                    )
                    data[filename_no_ext] = temp_data

                    # List of all 80 class names in COCO order
                    coco_classes = [
                        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
                        "boat", "traffic_light", "fire_hydrant", "stop_sign", "parking_meter", "bench",
                        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
                        "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
                        "skis", "snowboard", "sports_ball", "kite", "baseball_bat", "baseball_glove",
                        "skateboard", "surfboard", "tennis_racket", "bottle", "wine_glass", "cup",
                        "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
                        "broccoli", "carrot", "hot_dog", "pizza", "donut", "cake", "chair", "couch",
                        "potted_plant", "bed", "dining_table", "toilet", "tv", "laptop", "mouse",
                        "remote", "keyboard", "cellphone", "microwave", "oven", "toaster", "sink",
                        "refrigerator", "book", "clock", "vase", "scissors", "teddy_bear",
                        "hair_drier", "toothbrush",
                    ]

                    # --- Ensure all needed columns exist and are integer type ---
                    missing = [c for c in coco_classes if c not in df_mapping.columns]
                    if missing:
                        df_mapping = df_mapping.with_columns([pl.lit(0).cast(pl.Int64).alias(c) for c in missing])

                    df_mapping = df_mapping.with_columns([
                        pl.col(c).cast(pl.Int64, strict=False).fill_null(0).alias(c) for c in coco_classes
                    ])

                    # also ensure aggregated columns exist
                    if "total_time" not in df_mapping.columns:
                        df_mapping = df_mapping.with_columns(pl.lit(0).alias("total_time"))
                    if "total_crossing_detect" not in df_mapping.columns:
                        df_mapping = df_mapping.with_columns(pl.lit(0).cast(pl.Int64).alias("total_crossing_detect"))

                    # --- Count unique objects per yolo-id ---
                    counts_df = (
                        df.unique(subset=["yolo-id", "unique-id"])
                          .group_by("yolo-id")
                          .len()
                          .rename({"len": "count"})
                    )

                    id_to_count = {int(r["yolo-id"]): int(r["count"]) for r in counts_df.to_dicts()}
                    counters = {class_name: int(id_to_count.get(i, 0)) for i, class_name in enumerate(coco_classes)}

                    # --- Update df_mapping for the given video_city_id ---
                    df_mapping = df_mapping.with_columns([
                        pl.when(pl.col("id") == video_city_id)
                          .then(pl.col(class_name) + pl.lit(counters[class_name]))
                          .otherwise(pl.col(class_name))
                          .alias(class_name)
                        for class_name in coco_classes
                    ])

                    # Add duration of segment
                    time_video = duration.get_duration(df_mapping, video_id, int(start_index))
                    df_mapping = df_mapping.with_columns(
                        pl.when(pl.col("id") == video_city_id)
                          .then(pl.col("total_time") + pl.lit(time_video))
                          .otherwise(pl.col("total_time"))
                          .alias("total_time")
                    )

                    # Add total crossing detected
                    df_mapping = df_mapping.with_columns(
                        pl.when(pl.col("id") == video_city_id)
                          .then(pl.col("total_crossing_detect") + pl.lit(len(ids)).cast(pl.Int64))
                          .otherwise(pl.col("total_crossing_detect"))
                          .alias("total_crossing_detect")
                    )

                    # Aggregated values
                    speed_value = metrics.calculate_speed_of_crossing(
                        df_mapping,
                        df,
                        {filename_no_ext: temp_data},
                    )

                    if speed_value is not None:
                        for outer_key, inner_dict in speed_value.items():
                            all_speed.setdefault(outer_key, {}).update(inner_dict)

                    time_value = metrics.time_to_start_cross(
                        df_mapping,
                        df,
                        {filename_no_ext: temp_data},
                    )

                    if time_value is not None:
                        for outer_key, inner_dict in time_value.items():
                            all_time.setdefault(outer_key, {}).update(inner_dict)

        person_counter = df_mapping.select(pl.col("person").sum()).item()
        bicycle_counter = df_mapping.select(pl.col("bicycle").sum()).item()
        car_counter = df_mapping.select(pl.col("car").sum()).item()
        motorcycle_counter = df_mapping.select(pl.col("motorcycle").sum()).item()
        bus_counter = df_mapping.select(pl.col("bus").sum()).item()
        truck_counter = df_mapping.select(pl.col("truck").sum()).item()
        cellphone_counter = df_mapping.select(pl.col("cellphone").sum()).item()
        traffic_light_counter = df_mapping.select(pl.col("traffic_light").sum()).item()
        stop_sign_counter = df_mapping.select(pl.col("stop_sign").sum()).item()

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
        print("df_mapping:", type(df_mapping))
        pedestrian_cross_city = dataset_stats.pedestrian_cross_per_city(pedestrian_crossing_count, df_mapping)
        pedestrian_cross_city_all = dataset_stats.pedestrian_cross_per_city(pedestrian_crossing_count_all,
                                                                            df_mapping)

        logger.info("Calculating counts of crossing events in countries.")
        pedestrian_cross_country = dataset_stats.pedestrian_cross_per_country(pedestrian_cross_city, df_mapping)
        pedestrian_cross_country_all = dataset_stats.pedestrian_cross_per_country(pedestrian_cross_city_all,
                                                                                  df_mapping)

        # Jaywalking data
        logger.info("Calculating parameters for detection of jaywalking.")

        (
            crossings_with_traffic_equipment_city,
            crossings_without_traffic_equipment_city,
            total_duration_by_city,
            crossings_with_traffic_equipment_country,
            crossings_without_traffic_equipment_country,
            total_duration_by_country,
        ) = events.crossing_event_with_traffic_equipment(df_mapping, data)

        # ----------------------------------------------------------------------
        # Add city-level crossing counts for with and without traffic equipment
        # ----------------------------------------------------------------------

        # Ensure target columns exist (init as 0.0 or null; choose 0.0 for counts)
        needed_cols = [
            "with_trf_light_day_city",
            "with_trf_light_night_city",
            "without_trf_light_day_city",
            "without_trf_light_night_city",
        ]
        for c in needed_cols:
            if c not in df_mapping.columns:
                df_mapping = df_mapping.with_columns(pl.lit(0.0).alias(c))

        # Helper: treat missing/unknown state consistently
        def _is_missing_state_expr(col: str) -> pl.Expr:
            return pl.col(col).is_null() | (pl.col(col).cast(pl.Utf8,
                                                             strict=False).str.to_lowercase().is_in(["nan",
                                                                                                     "na",
                                                                                                     "n/a",
                                                                                                     "unknown",
                                                                                                     ""]))

        # ---- WITH equipment ----
        with_rows = []
        for key, value in crossings_with_traffic_equipment_city.items():
            parts = key.split("_")
            if len(parts) < 4:
                continue
            city = parts[0]
            lat = float(parts[1])
            lon = float(parts[2])
            tod = int(parts[3])  # 0 day, 1 night
            with_rows.append({"city": city, "lat": lat, "lon": lon, "_tod": tod, "_val": float(value)})

        if with_rows:
            df_with = pl.DataFrame(with_rows)

            # bring in state from mapping (first non-null state for city+lat+lon, else null)
            if "state" in df_mapping.columns:
                state_lookup = (
                    df_mapping
                    .select(["city", "lat", "lon", "state"])
                    .group_by(["city", "lat", "lon"])
                    .agg(pl.col("state").drop_nulls().first().alias("_state"))
                )
                df_with = df_with.join(state_lookup, on=["city", "lat", "lon"], how="left")
            else:
                df_with = df_with.with_columns(pl.lit(None).cast(pl.Utf8).alias("_state"))

            df_with = df_with.with_columns([
                pl.when(pl.col("_tod") == 0).then(pl.col("_val")).otherwise(None).alias("_with_day"),
                pl.when(pl.col("_tod") == 1).then(pl.col("_val")).otherwise(None).alias("_with_night"),
            ]).group_by(["city", "lat", "lon", "_state"]).agg([
                pl.col("_with_day").drop_nulls().last().alias("with_trf_light_day_city"),
                pl.col("_with_night").drop_nulls().last().alias("with_trf_light_night_city"),
            ])

            # Join updates to mapping by city+lat+lon, then only apply if state matches OR both missing
            df_mapping = df_mapping.join(
                df_with.select(["city", "lat", "lon", "_state",
                                "with_trf_light_day_city", "with_trf_light_night_city"]),
                on=["city", "lat", "lon"],
                how="left",
                suffix="_upd",
            )

            if "state" in df_mapping.columns:
                state_match = (
                    (pl.col("state").cast(pl.Utf8, strict=False) == pl.col("_state").cast(pl.Utf8, strict=False))
                    | (_is_missing_state_expr("state") & _is_missing_state_expr("_state"))
                )
            else:
                state_match = pl.lit(True)

            df_mapping = df_mapping.with_columns([
                pl.when(state_match & pl.col("with_trf_light_day_city_upd").is_not_null())
                  .then(pl.col("with_trf_light_day_city_upd"))
                  .otherwise(pl.col("with_trf_light_day_city"))
                  .alias("with_trf_light_day_city"),
                pl.when(state_match & pl.col("with_trf_light_night_city_upd").is_not_null())
                  .then(pl.col("with_trf_light_night_city_upd"))
                  .otherwise(pl.col("with_trf_light_night_city"))
                  .alias("with_trf_light_night_city"),
            ]).drop(["with_trf_light_day_city_upd", "with_trf_light_night_city_upd", "_state"])

        # ---- WITHOUT equipment ----
        without_rows = []
        for key, value in crossings_without_traffic_equipment_city.items():
            parts = key.split("_")
            if len(parts) < 4:
                continue
            city = parts[0]
            lat = float(parts[1])
            lon = float(parts[2])
            tod = int(parts[3])
            without_rows.append({"city": city, "lat": lat, "lon": lon, "_tod": tod, "_val": float(value)})

        if without_rows:
            df_wo = pl.DataFrame(without_rows)

            if "state" in df_mapping.columns:
                state_lookup = (
                    df_mapping
                    .select(["city", "lat", "lon", "state"])
                    .group_by(["city", "lat", "lon"])
                    .agg(pl.col("state").drop_nulls().first().alias("_state"))
                )
                df_wo = df_wo.join(state_lookup, on=["city", "lat", "lon"], how="left")
            else:
                df_wo = df_wo.with_columns(pl.lit(None).cast(pl.Utf8).alias("_state"))

            df_wo = df_wo.with_columns([
                pl.when(pl.col("_tod") == 0).then(pl.col("_val")).otherwise(None).alias("_wo_day"),
                pl.when(pl.col("_tod") == 1).then(pl.col("_val")).otherwise(None).alias("_wo_night"),
            ]).group_by(["city", "lat", "lon", "_state"]).agg([
                pl.col("_wo_day").drop_nulls().last().alias("without_trf_light_day_city"),
                pl.col("_wo_night").drop_nulls().last().alias("without_trf_light_night_city"),
            ])

            df_mapping = df_mapping.join(
                df_wo.select(["city",
                              "lat",
                              "lon",
                              "_state",
                              "without_trf_light_day_city",
                              "without_trf_light_night_city"]),

                on=["city", "lat", "lon"],
                how="left",
                suffix="_upd",
            )

            if "state" in df_mapping.columns:
                state_match = (
                    (pl.col("state").cast(pl.Utf8, strict=False) == pl.col("_state").cast(pl.Utf8, strict=False))
                    | (_is_missing_state_expr("state") & _is_missing_state_expr("_state"))
                )
            else:
                state_match = pl.lit(True)

            df_mapping = df_mapping.with_columns([
                pl.when(state_match & pl.col("without_trf_light_day_city_upd").is_not_null())
                  .then(pl.col("without_trf_light_day_city_upd"))
                  .otherwise(pl.col("without_trf_light_day_city"))
                  .alias("without_trf_light_day_city"),
                pl.when(state_match & pl.col("without_trf_light_night_city_upd").is_not_null())
                  .then(pl.col("without_trf_light_night_city_upd"))
                  .otherwise(pl.col("without_trf_light_night_city"))
                  .alias("without_trf_light_night_city"),
            ]).drop(["without_trf_light_day_city_upd", "without_trf_light_night_city_upd", "_state"])

        # ----------------------------------------------------------------------
        # Add country-level crossing counts for with and without traffic equipment
        # ----------------------------------------------------------------------

        # Ensure target columns exist
        needed_country_cols = [
            "with_trf_light_day_country",
            "with_trf_light_night_country",
            "without_trf_light_day_country",
            "without_trf_light_night_country",
        ]
        for c in needed_country_cols:
            if c not in df_mapping.columns:
                df_mapping = df_mapping.with_columns(pl.lit(0.0).alias(c))

        # Build updates for "with"
        with_country_rows = []
        for key, value in crossings_with_traffic_equipment_country.items():
            parts = key.split("_")
            if len(parts) < 2:
                continue
            country = parts[0]
            tod = int(parts[1])  # 0 day, 1 night
            with_country_rows.append({"country": country, "_tod": tod, "_val": float(value)})

        if with_country_rows:
            df_with_country = (
                pl.DataFrame(with_country_rows)
                .with_columns([
                    pl.when(pl.col("_tod") == 0).then(pl.col("_val")).otherwise(None)
                    .alias("with_trf_light_day_country"),
                    pl.when(pl.col("_tod") == 1).then(pl.col("_val")).otherwise(None)
                    .alias("with_trf_light_night_country"),
                ])
                .group_by("country")
                .agg([
                    pl.col("with_trf_light_day_country").drop_nulls().last().alias("with_trf_light_day_country"),
                    pl.col("with_trf_light_night_country").drop_nulls().last().alias("with_trf_light_night_country"),
                ])
            )

            df_mapping = (
                df_mapping
                .join(df_with_country, on="country", how="left", suffix="_upd")
                .with_columns([
                    pl.coalesce([pl.col("with_trf_light_day_country_upd"), pl.col("with_trf_light_day_country")])
                      .alias("with_trf_light_day_country"),
                    pl.coalesce([pl.col("with_trf_light_night_country_upd"), pl.col("with_trf_light_night_country")])
                      .alias("with_trf_light_night_country"),
                ])
                .drop(["with_trf_light_day_country_upd", "with_trf_light_night_country_upd"])
            )

        # Build updates for "without"
        without_country_rows = []
        for key, value in crossings_without_traffic_equipment_country.items():
            parts = key.split("_")
            if len(parts) < 2:
                continue
            country = parts[0]
            tod = int(parts[1])
            without_country_rows.append({"country": country, "_tod": tod, "_val": float(value)})

        if without_country_rows:
            df_without_country = (
                pl.DataFrame(without_country_rows)
                .with_columns([
                    pl.when(pl.col("_tod") == 0).then(pl.col("_val")).otherwise(None)
                    .alias("without_trf_light_day_country"),
                    pl.when(pl.col("_tod") == 1).then(pl.col("_val")).otherwise(None)
                    .alias("without_trf_light_night_country"),
                ])
                .group_by("country")
                .agg([
                    pl.col("without_trf_light_day_country").drop_nulls().last()
                    .alias("without_trf_light_day_country"),
                    pl.col("without_trf_light_night_country").drop_nulls().last()
                    .alias("without_trf_light_night_country"),
                ])
            )

            df_mapping = (
                df_mapping
                .join(df_without_country, on="country", how="left", suffix="_upd")
                .with_columns([
                    pl.coalesce([pl.col("without_trf_light_day_country_upd"),
                                 pl.col("without_trf_light_day_country")])
                      .alias("without_trf_light_day_country"),
                    pl.coalesce([pl.col("without_trf_light_night_country_upd"),
                                 pl.col("without_trf_light_night_country")])
                      .alias("without_trf_light_night_country"),
                ])
                .drop(["without_trf_light_day_country_upd", "without_trf_light_night_country_upd"])
            )

        # ---------------------------------------
        # Add city-level crossing counts detected
        # ---------------------------------------

        # Ensure city-level columns exist
        needed_city_cols = [
            "crossing_detected_city_day",
            "crossing_detected_city_night",
            "crossing_detected_city_all_day",
            "crossing_detected_city_all_night",
            "crossing_detected_city",
            "crossing_detected_city_all",
        ]
        for c in needed_city_cols:
            if c not in df_mapping.columns:
                df_mapping = df_mapping.with_columns(pl.lit(0.0).alias(c))

        # City day/night (filtered ids)
        rows_city = []
        for city_lat_long_cond, value in pedestrian_cross_city.items():
            parts = city_lat_long_cond.split("_")
            if len(parts) < 4:
                continue
            city, lat_s, lon_s, cond = parts[0], parts[1], parts[2], parts[3]
            try:
                lat = float(lat_s)
            except Exception:
                continue
            if cond == "0":
                rows_city.append({"city": city, "lat": lat, "crossing_detected_city_day": float(value),
                                  "crossing_detected_city_night": None})
            elif cond == "1":
                rows_city.append({"city": city, "lat": lat, "crossing_detected_city_day": None,
                                  "crossing_detected_city_night": float(value)})

        if rows_city:
            upd_city = (
                pl.DataFrame(rows_city)
                .group_by(["city", "lat"])
                .agg([
                    pl.col("crossing_detected_city_day").drop_nulls().last().alias("crossing_detected_city_day"),
                    pl.col("crossing_detected_city_night").drop_nulls().last().alias("crossing_detected_city_night"),
                ])
            )

            df_mapping = (
                df_mapping
                .join(upd_city, on=["city", "lat"], how="left", suffix="_upd")
                .with_columns([
                    pl.coalesce([pl.col("crossing_detected_city_day_upd"), pl.col("crossing_detected_city_day")])
                      .alias("crossing_detected_city_day"),
                    pl.coalesce([pl.col("crossing_detected_city_night_upd"), pl.col("crossing_detected_city_night")])
                      .alias("crossing_detected_city_night"),
                ])
                .drop(["crossing_detected_city_day_upd", "crossing_detected_city_night_upd"])
            )

        # City day/night (all ids)
        rows_city_all = []
        for city_lat_long_cond, value in pedestrian_cross_city_all.items():
            parts = city_lat_long_cond.split("_")
            if len(parts) < 4:
                continue
            city, lat_s, lon_s, cond = parts[0], parts[1], parts[2], parts[3]
            try:
                lat = float(lat_s)
            except Exception:
                continue
            if cond == "0":
                rows_city_all.append({"city": city, "lat": lat, "crossing_detected_city_all_day": float(value),
                                      "crossing_detected_city_all_night": None})
            elif cond == "1":
                rows_city_all.append({"city": city, "lat": lat, "crossing_detected_city_all_day": None,
                                      "crossing_detected_city_all_night": float(value)})

        if rows_city_all:
            upd_city_all = (
                pl.DataFrame(rows_city_all)
                .group_by(["city", "lat"])
                .agg([
                    pl.col("crossing_detected_city_all_day").drop_nulls().last()
                    .alias("crossing_detected_city_all_day"),
                    pl.col("crossing_detected_city_all_night").drop_nulls().last()
                    .alias("crossing_detected_city_all_night"),
                ])
            )

            df_mapping = (
                df_mapping
                .join(upd_city_all, on=["city", "lat"], how="left", suffix="_upd")
                .with_columns([
                    pl.coalesce([pl.col("crossing_detected_city_all_day_upd"),
                                 pl.col("crossing_detected_city_all_day")])
                      .alias("crossing_detected_city_all_day"),
                    pl.coalesce([pl.col("crossing_detected_city_all_night_upd"),
                                 pl.col("crossing_detected_city_all_night")])
                      .alias("crossing_detected_city_all_night"),
                ])
                .drop(["crossing_detected_city_all_day_upd", "crossing_detected_city_all_night_upd"])
            )

        # Totals (same semantics as fillna(0) + fillna(0))
        df_mapping = df_mapping.with_columns([
            (pl.col("crossing_detected_city_day").fill_null(0.0) + pl.col("crossing_detected_city_night")
             .fill_null(0.0)).alias("crossing_detected_city"),
            (pl.col("crossing_detected_city_all_day").fill_null(0.0) + pl.col("crossing_detected_city_all_night")
             .fill_null(0.0)).alias("crossing_detected_city_all"),
        ])

        # ---------------------------------------
        # Add country-level crossing counts detected
        # ---------------------------------------

        # Ensure columns exist (init as 0.0)
        needed_country_detect_cols = [
            "crossing_detected_country_day",
            "crossing_detected_country_night",
            "crossing_detected_country_all_day",
            "crossing_detected_country_all_night",
            "crossing_detected_country",
            "crossing_detected_country_all",
        ]
        for c in needed_country_detect_cols:
            if c not in df_mapping.columns:
                df_mapping = df_mapping.with_columns(pl.lit(0.0).alias(c))

        # Updates: filtered ids
        rows_country = []
        for country_cond, value in pedestrian_cross_country.items():
            try:
                country, cond = country_cond.rsplit("_", 1)
            except ValueError:
                continue

            if cond == "0":
                rows_country.append({"country": country, "crossing_detected_country_day": float(value),
                                     "crossing_detected_country_night": None})
            elif cond == "1":
                rows_country.append({"country": country, "crossing_detected_country_day": None,
                                     "crossing_detected_country_night": float(value)})

        if rows_country:
            upd_country = (
                pl.DataFrame(rows_country)
                .group_by("country")
                .agg([
                    pl.col("crossing_detected_country_day").drop_nulls().last()
                    .alias("crossing_detected_country_day"),
                    pl.col("crossing_detected_country_night").drop_nulls().last()
                    .alias("crossing_detected_country_night"),
                ])
            )

            df_mapping = (
                df_mapping
                .join(upd_country, on="country", how="left", suffix="_upd")
                .with_columns([
                    pl.coalesce([pl.col("crossing_detected_country_day_upd"),
                                 pl.col("crossing_detected_country_day")])
                      .alias("crossing_detected_country_day"),
                    pl.coalesce([pl.col("crossing_detected_country_night_upd"),
                                 pl.col("crossing_detected_country_night")])
                      .alias("crossing_detected_country_night"),
                ])
                .drop(["crossing_detected_country_day_upd", "crossing_detected_country_night_upd"])
            )

        # Updates: all ids
        rows_country_all = []
        for country_cond, value in pedestrian_cross_country_all.items():
            try:
                country, cond = country_cond.rsplit("_", 1)
            except ValueError:
                continue

            if cond == "0":
                rows_country_all.append({"country": country, "crossing_detected_country_all_day": float(value),
                                         "crossing_detected_country_all_night": None})
            elif cond == "1":
                rows_country_all.append({"country": country, "crossing_detected_country_all_day": None,
                                         "crossing_detected_country_all_night": float(value)})

        if rows_country_all:
            upd_country_all = (
                pl.DataFrame(rows_country_all)
                .group_by("country")
                .agg([
                    pl.col("crossing_detected_country_all_day").drop_nulls().last()
                    .alias("crossing_detected_country_all_day"),
                    pl.col("crossing_detected_country_all_night").drop_nulls().last()
                    .alias("crossing_detected_country_all_night"),
                ])
            )

            df_mapping = (
                df_mapping
                .join(upd_country_all, on="country", how="left", suffix="_upd")
                .with_columns([
                    pl.coalesce([pl.col("crossing_detected_country_all_day_upd"),
                                 pl.col("crossing_detected_country_all_day")])
                      .alias("crossing_detected_country_all_day"),
                    pl.coalesce([pl.col("crossing_detected_country_all_night_upd"),
                                 pl.col("crossing_detected_country_all_night")])
                      .alias("crossing_detected_country_all_night"),
                ])
                .drop(["crossing_detected_country_all_day_upd", "crossing_detected_country_all_night_upd"])
            )

        # Totals
        df_mapping = df_mapping.with_columns([
            (pl.col("crossing_detected_country_day").fill_null(0.0) + pl.col("crossing_detected_country_night")
             .fill_null(0.0)).alias("crossing_detected_country"),
            (pl.col("crossing_detected_country_all_day").fill_null(0.0) + pl.col("crossing_detected_country_all_night")
             .fill_null(0.0)).alias("crossing_detected_country_all"),
        ])

        # Add column with count of videos
        # Same semantics as: len(x.strip("[]").split(",")) if x.strip("[]") else 0
        videos_stripped = pl.col("videos").cast(pl.Utf8, strict=False).str.strip_chars("[]").str.strip_chars()
        df_mapping = df_mapping.with_columns(
            pl.when(pl.col("videos").is_null() | (videos_stripped == ""))
              .then(0)
              .otherwise(videos_stripped.str.split(",").list.len())
              .alias("total_videos")
        )

        # Get lat and lon for cities (row-wise; geopy is inherently scalar)
        logger.info("Fetching lat and lon coordinates for cities.")

        # Ensure columns exist
        for c in ["lat", "lon"]:
            if c not in df_mapping.columns:
                df_mapping = df_mapping.with_columns(pl.lit(None).cast(pl.Float64).alias(c))

        lat_list = df_mapping.get_column("lat").to_list()
        lon_list = df_mapping.get_column("lon").to_list()

        city_list = df_mapping.get_column("city").to_list()
        state_list = df_mapping.get_column("state").to_list() if "state" in df_mapping.columns else [None] * df_mapping.height  # noqa: E501
        country_list = df_mapping.get_column("country").to_list()

        for i in tqdm(range(df_mapping.height), total=df_mapping.height):
            lat = lat_list[i]
            lon = lon_list[i]
            # Treat None as missing; also treat NaN floats as missing
            lat_missing = (lat is None) or (isinstance(lat, float) and lat != lat)
            lon_missing = (lon is None) or (isinstance(lon, float) and lon != lon)

            if lat_missing or lon_missing:
                new_lat, new_lon = geo.get_coordinates(city_list[i], state_list[i], common.correct_country(country_list[i]))  # type: ignore  # noqa: E501
                lat_list[i] = new_lat
                lon_list[i] = new_lon

        df_mapping = df_mapping.with_columns([
            pl.Series("lat", lat_list).cast(pl.Float64, strict=False),
            pl.Series("lon", lon_list).cast(pl.Float64, strict=False),
        ])

        # Save the raw file for further investigation
        df_mapping_raw = df_mapping.clone()

        # ----------------------------------------------------------------------
        # df_mapping_raw cleanup + save
        # ----------------------------------------------------------------------

        drop_cols = ['gmp', 'population_city', 'population_country', 'traffic_mortality',
                     'literacy_rate', 'avg_height', 'med_age', 'gini', 'traffic_index', 'videos',
                     'time_of_day', 'start_time', 'end_time', 'vehicle_type', 'upload_date']

        df_mapping_raw = df_mapping_raw.drop([c for c in drop_cols if c in df_mapping_raw.columns])

        # apply tools.count_unique_channels to channel column
        if "channel" in df_mapping_raw.columns:
            df_mapping_raw = df_mapping_raw.with_columns(
                pl.col("channel").map_elements(
                    tools.count_unique_channels,
                    return_dtype=pl.Int64,  # adjust if your function returns float/str
                ).alias("channel")
            )

        df_mapping_raw.write_csv(os.path.join(common.output_dir, "mapping_city_raw.csv"))

        # ----------------------------------------------------------------------
        # Filters (population thresholds)
        # ----------------------------------------------------------------------

        population_threshold = common.get_configs("population_threshold")
        min_percentage = common.get_configs("min_city_population_percentage")

        # Ensure numeric types (coerce invalid to null)
        df_mapping = df_mapping.with_columns([
            pl.col("population_city").cast(pl.Float64, strict=False).alias("population_city"),
            pl.col("population_country").cast(pl.Float64, strict=False).alias("population_country"),
        ])

        df_mapping = df_mapping.filter(
            (pl.col("population_city") >= float(population_threshold)) |
            (pl.col("population_city") >= float(min_percentage) * pl.col("population_country"))
        )

        # Remove the rows of the cities where the footage recorded is less than threshold
        df_mapping = dataset_stats.remove_columns_below_threshold(df_mapping, common.get_configs("footage_threshold"))

        # Limit countries if required
        countries_include = common.get_configs("countries_analyse")
        if countries_include:
            df_mapping = df_mapping.filter(pl.col("iso3").is_in(countries_include))

        total_duration = dataset_stats.calculate_total_seconds(df_mapping)

        logger.info(
            f"Duration of videos in seconds after filtering: {total_duration}, in"
            f" minutes after filtering: {total_duration/60:.2f}, in "
            f"hours: {total_duration/60/60:.2f}."
        )

        logger.info("Total number of videos after filtering: {}.",
                    dataset_stats.calculate_total_videos(df_mapping))

        country, number, _ = metrics_cache.get_unique_values(df_mapping, "iso3")
        logger.info(f"Total number of countries and territories after filtering: {number}.")

        city_state_iso3, number, dup_report = metrics_cache.get_unique_values(
            df_mapping,
            ["city", "state", "iso3"],
            return_duplicates=True,
        )

        logger.info(f"Total number of unique city+state+iso3 keys after filtering: {number}.")

        if dup_report is not None and dup_report.height > 0:
            logger.warning(f"Duplicated keys:\n{dup_report}")

        df_mapping = mapping_enrich.add_speed_and_time_to_mapping(
            df_mapping=df_mapping,
            avg_speed_city=avg_speed_city,
            avg_speed_country=avg_speed_country,
            avg_time_city=avg_time_city,
            avg_time_country=avg_time_country,
            pedestrian_cross_city=pedestrian_cross_city,
            pedestrian_cross_country=pedestrian_cross_country,
        )

        min_max_speed = duration.get_duration_segment(all_speed, df_mapping, name="speed", duration=None)
        min_max_time = duration.get_duration_segment(all_time, df_mapping, name="time", duration=None)

        # Save the results to a pickle file
        logger.info("Saving results to a pickle file {}.", file_results)
        with open(file_results, "wb") as file:
            pickle.dump(
                (
                    data,                                         # 0
                    person_counter,                               # 1
                    bicycle_counter,                              # 2
                    car_counter,                                  # 3
                    motorcycle_counter,                           # 4
                    bus_counter,                                  # 5
                    truck_counter,                                # 6
                    cellphone_counter,                            # 7
                    traffic_light_counter,                        # 8
                    stop_sign_counter,                            # 9
                    pedestrian_cross_city,                        # 10
                    pedestrian_crossing_count,                    # 11
                    person_city,                                  # 12
                    bicycle_city,                                 # 13
                    car_city,                                     # 14
                    motorcycle_city,                              # 15
                    bus_city,                                     # 16
                    truck_city,                                   # 17
                    cross_evnt_city,                              # 18
                    vehicle_city,                                 # 19
                    cellphone_city,                               # 20
                    traffic_sign_city,                            # 21
                    all_speed,                                    # 22
                    all_time,                                     # 23
                    avg_time_city,                                # 24
                    avg_speed_city,                               # 25
                    df_mapping,                                   # 26
                    avg_speed_country,                            # 27
                    avg_time_country,                             # 28
                    crossings_with_traffic_equipment_city,        # 29
                    crossings_without_traffic_equipment_city,     # 30
                    crossings_with_traffic_equipment_country,     # 31
                    crossings_without_traffic_equipment_country,  # 32
                    min_max_speed,                                # 33
                    min_max_time,                                 # 34
                    pedestrian_cross_country,                     # 35
                    all_speed_city,                               # 36
                    all_time_city,                                # 37
                    all_speed_country,                            # 38
                    all_time_country,                             # 39
                    df_mapping_raw,                               # 40
                    pedestrian_cross_city_all,                    # 41
                    pedestrian_cross_country_all,                 # 42
                ),
                file,
            )

        logger.info("Analysis results saved to pickle file.")

    # Set index as ID  (Polars has no index; keep semantics by ensuring `id` exists and is first)
    if "id" in df_mapping.columns:
        df_mapping = df_mapping.select(["id"] + [c for c in df_mapping.columns if c != "id"])

        # --- Check if reanalysis of speed is required ---
    if common.get_configs("reanalyse_speed"):
        # NOTE: if your Metrics/Mapping_Enrich now expect polars, keep as-is;
        # if any still expects pandas, convert inside those functions (not here).
        avg_speed_country = metrics.avg_speed_of_crossing_country(df_mapping, all_speed)
        avg_speed_city = metrics.avg_speed_of_crossing_city(df_mapping, all_speed)
        df_mapping = mapping_enrich.add_speed_and_time_to_mapping(
            df_mapping=df_mapping,
            avg_speed_city=avg_speed_city,
            avg_time_city=None,
            avg_speed_country=avg_speed_country,
            avg_time_country=None,
            pedestrian_cross_city=pedestrian_cross_city,
            pedestrian_cross_country=pedestrian_cross_country,
        )
        # --- Update avg speed values in the pickle file ---
        with open(file_results, "rb") as file:
            results = pickle.load(file)
        results_list = list(results)
        results_list[25] = avg_speed_city     # Update city speed
        results_list[27] = avg_speed_country  # Update country speed
        results_list[26] = df_mapping         # Update mapping (polars)
        with open(file_results, "wb") as file:
            pickle.dump(tuple(results_list), file)
        logger.info("Updated speed values in the pickle file.")

    # --- Check if reanalysis of waiting time is required ---
    if common.get_configs("reanalyse_waiting_time"):
        avg_time_country = metrics.avg_time_to_start_cross_country(df_mapping, all_time)
        avg_time_city = metrics.avg_time_to_start_cross_city(df_mapping, all_time)
        df_mapping = mapping_enrich.add_speed_and_time_to_mapping(
            df_mapping=df_mapping,
            avg_time_city=avg_time_city,
            avg_speed_city=avg_speed_city,
            avg_time_country=avg_time_country,
            avg_speed_country=avg_speed_country,
            pedestrian_cross_city=pedestrian_cross_city,
            pedestrian_cross_country=pedestrian_cross_country,
        )
        # --- Update avg time values in the pickle file ---
        with open(file_results, "rb") as file:
            results = pickle.load(file)
        results_list = list(results)
        results_list[24] = avg_time_city     # Update city waiting time
        results_list[28] = avg_time_country  # Update country waiting time
        results_list[26] = df_mapping        # Update mapping
        with open(file_results, "wb") as file:
            pickle.dump(tuple(results_list), file)
        logger.info("Updated time values in the pickle file.")

    # --- Remove countries/cities with insufficient crossing detections ---
    if common.get_configs("min_crossing_detect") != 0:
        threshold: float = float(common.get_configs("min_crossing_detect"))
        country_detect: Dict[str, Dict[str, float]] = {}
        for key, value in pedestrian_cross_country.items():
            country, cond = key.rsplit("_", 1)
            val_f = float(value)
            if country not in country_detect:
                country_detect[country] = {}
            country_detect[country][cond] = val_f

        # Find countries where BOTH conditions are below threshold
        keep_countries: Set[str] = {
            country for country, vals in country_detect.items()
            if (("0" in vals or "1" in vals) and (vals.get("0", 0.0) + vals.get("1", 0.0) >= threshold))
        }

        if keep_countries:
            df_mapping = df_mapping.filter(pl.col("country").is_in(list(keep_countries)))
        else:
            df_mapping = df_mapping.head(0)  # no countries meet threshold

    # Sort by continent and city, both in ascending order
    df_mapping = df_mapping.sort(["continent", "city"])

    # Save updated mapping file in output
    os.makedirs(common.output_dir, exist_ok=True)
    df_mapping.write_csv(os.path.join(common.output_dir, "mapping_updated.csv"))

    logger.info("Detected:")
    logger.info(f"person: {person_counter}; bicycle: {bicycle_counter}; car: {car_counter}")
    logger.info(f"motorcycle: {motorcycle_counter}; bus: {bus_counter}; truck: {truck_counter}")
    logger.info(f"cellphone: {cellphone_counter}; traffic light: {traffic_light_counter}; " +
                f"traffic sign: {stop_sign_counter}")
    logger.info("Producing output.")

    # Data to avoid showing on hover in scatter plots
    columns_remove = ['videos', 'time_of_day', 'start_time', 'end_time', 'upload_date', 'fps_list', 'vehicle_type',
                      'channel']

    hover_data = [c for c in df_mapping.columns if c not in set(columns_remove)]
    df = df_mapping.clone()  # copy df to manipulate for output

    # Set state to NA (Polars nulls)
    if "state" in df.columns:
        df = df.with_columns(pl.col("state").fill_null("NA").alias("state"))

    # Maps with filtered data
    maps.mapbox_map(df=df.to_pandas(), hover_data=hover_data, file_name='mapbox_map')

    maps.mapbox_map(df=df.to_pandas(),
                    hover_data=hover_data,
                    density_col='total_time',
                    density_radius=10,
                    file_name='mapbox_map_time')

    maps.world_map(df_mapping=df.to_pandas())  # map with countries

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
        df = (df_mapping.filter(
                (pl.col("speed_crossing") != 0) &
                (pl.col("time_crossing") != 0)
            ).with_columns(
                pl.col("state").fill_null("NA").alias("state")))

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

        # ---------------------------------------------------------------------
        # Hover columns (keep order stable; exclude configured columns)
        # ---------------------------------------------------------------------
        columns_remove = ['videos', 'time_of_day', 'start_time', 'end_time', 'upload_date', 'fps_list', 'vehicle_type']
        remove_set = set(columns_remove)
        hover_data = [c for c in df_countries.columns if c not in remove_set]

        columns_remove_raw = ['gini', 'traffic_mortality', 'avg_height', 'population_country', 'population_city',
                              'med_age', 'literacy_rate']
        remove_raw_set = remove_set | set(columns_remove_raw)
        hover_data_raw = [c for c in df_countries.columns if c not in remove_raw_set]

        # ---------------------------------------------------------------------
        # Save via pandas (handles nested data by serializing to string/object)
        # ---------------------------------------------------------------------
        os.makedirs(common.output_dir, exist_ok=True)
        df_countries.to_pandas().to_csv(os.path.join(common.output_dir, "mapping_countries.csv"), index=False)

        # Map with images. currently works on a 13" MacBook air screen in chrome, as things are hardcoded...
        maps.map_world(df=df_countries_raw.to_pandas(),
                       color="continent",                # same default as map_political
                       show_cities=True,
                       df_cities=df_mapping.to_pandas(),
                       show_images=True,
                       hover_data=hover_data_raw,
                       save_file=True,
                       save_final=False,
                       file_name="raw_map")

        # Map with screenshots and countries colours by continent
        maps.map_world(df=df_countries.to_pandas(),
                       color="continent",
                       show_cities=True,
                       df_cities=df_mapping.to_pandas(),
                       show_images=True,
                       hover_data=hover_data,
                       save_file=False,
                       save_final=False,
                       file_name="map_screenshots",
                       show_colorbar=True,
                       colorbar_title="Continent",
                       colorbar_kwargs=dict(y=0.035, len=0.55, bgcolor="rgba(255,255,255,0.9)"))

        # Map with screenshots and countries colours by amount of footage
        remove_set = set(columns_remove)
        hover_data = [c for c in df_countries_raw.columns if c not in remove_set]

        # log(1 + x) to avoid -inf for zero
        df_countries_raw = df_countries_raw.with_columns(
            pl.col("total_time")
              .fill_null(0)
              .cast(pl.Float64)
              .log1p()
              .alias("log_total_time")
        )

        # Produce map with all data
        df = df_mapping_raw.clone()  # copy df to manipulate for output
        df = df.with_columns(
            pl.col("state").fill_null("NA").alias("state")
        )

        # Sort by continent and city, both in ascending order
        df = df.sort(["continent", "city"])

        maps.map_world(df=df_countries_raw.to_pandas(),
                       color="log_total_time",
                       show_cities=True,
                       df_cities=df_mapping.to_pandas(),
                       show_images=True,
                       hover_data=hover_data,
                       show_colorbar=True,
                       colorbar_title="Footage (log)",
                       save_file=True,
                       save_final=False,
                       file_name="map_screenshots_total_time")

        # Drop columns (only if present)
        drop_cols = [
            "speed_crossing_day_country",
            "speed_crossing_night_country",
            "speed_crossing_day_night_country_avg",
            "time_crossing_day_country",
            "time_crossing_night_country",
            "time_crossing_day_night_country_avg",
        ]
        df_countries_raw = df_countries_raw.drop([c for c in drop_cols if c in df_countries_raw.columns])

        # Save via pandas (CSV-friendly for nested columns)
        os.makedirs(common.output_dir, exist_ok=True)
        df_countries_raw.to_pandas().to_csv(
            os.path.join(common.output_dir, "mapping_countries_raw.csv"),
            index=False
        )
        print(df_countries.head())
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
        df = (df_countries.filter(pl.col("person") != 0).with_columns(
                (pl.col("person") / pl.col("total_time")).alias("person_norm")))

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
        df = (df_countries.filter(pl.col("bicycle") != 0).with_columns(
                (pl.col("bicycle") / pl.col("total_time")).alias("bicycle_norm")))

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
        df = (
            df_countries
            .filter(pl.col("speed_crossing_day_night_country_avg") != 0)
            .filter(pl.col("time_crossing_day_night_country_avg") != 0)
        )
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
        df = (df_countries
              .filter(pl.col("speed_crossing_day_country") != 0)
              .filter(pl.col("time_crossing_day_country") != 0))
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
        df = (df_countries
              .filter(pl.col("speed_crossing_night_country") != 0)
              .filter(pl.col("time_crossing_night_country") != 0))
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
        df = (df_countries
              .filter(pl.col("time_crossing_day_night_country_avg") != 0)
              .filter(pl.col("population_country").is_not_null() & (pl.col("population_country") != 0)))
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
        df = (df_countries
              .filter(pl.col("speed_crossing_day_night_country_avg") != 0)
              .filter(pl.col("population_country").is_not_null() & (pl.col("population_country") != 0)))
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
        df = (df_countries
              .filter(pl.col("time_crossing_day_night_country_avg") != 0)
              .filter(pl.col("traffic_mortality").is_not_null() & (pl.col("traffic_mortality") != 0)))
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
        df = (df_countries
              .filter(pl.col("speed_crossing_day_night_country_avg") != 0)
              .filter(pl.col("traffic_mortality").is_not_null() & (pl.col("traffic_mortality") != 0)))
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
        df = (df_countries
              .filter(pl.col("time_crossing_day_night_country_avg") != 0)
              .filter(pl.col("literacy_rate").is_not_null() & (pl.col("literacy_rate") != 0)))
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
        df = (df_countries
              .filter(pl.col("speed_crossing_day_night_country_avg") != 0)
              .filter(pl.col("literacy_rate").is_not_null() & (pl.col("literacy_rate") != 0)))
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
        df = (df_countries
              .filter(pl.col("time_crossing_day_night_country_avg") != 0)
              .filter(pl.col("gini").is_not_null() & (pl.col("gini") != 0)))
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
        df = (df_countries
              .filter(pl.col("speed_crossing_day_night_country_avg") != 0)
              .filter(pl.col("gini").is_not_null() & (pl.col("gini") != 0)))
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
        df = (df_countries
              .filter(pl.col("time_crossing_day_night_country_avg") != 0)
              .filter(pl.col("med_age").is_not_null() & (pl.col("med_age") != 0)))
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
        df = (df_countries
              .filter(pl.col("speed_crossing_day_night_country_avg") != 0)
              .filter(pl.col("med_age") != 0))
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
        df = (df_countries
              .filter(pl.col("time_crossing_day_night_country_avg") != 0)
              .with_columns(
                (pl.col("cellphone") / pl.col("total_time")).alias("cellphone_normalised")))
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
        df = (df_countries
              .filter(pl.col("speed_crossing_day_night_country_avg") != 0)
              .with_columns(
                (pl.col("cellphone") / pl.col("total_time")).alias("cellphone_normalised")))
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
        maps.map_world(df=df_countries.to_pandas(),
                       color="speed_crossing_day_night_country_avg",
                       title="Mean speed of crossing (in m/s)",
                       show_colorbar=True,
                       colorbar_title="",                 # keep your empty title behavior
                       filter_zero_nan=True,              # preserves old map() filtering
                       save_file=True,
                       file_name="map_speed_crossing"
                       )

        # Crossing initiation time (used to be plots_class.map)
        maps.map_world(df=df_countries.to_pandas(),
                       color="time_crossing_day_night_country_avg",
                       title="Crossing initiation time (in s)",
                       show_colorbar=True,
                       colorbar_title="",
                       filter_zero_nan=True,
                       save_file=True,
                       file_name="map_crossing_time"
                       )

        # Crossing with and without traffic lights
        df = (df_countries.with_columns([
                ((pl.col("with_trf_light_day_country") + pl.col("with_trf_light_night_country"))
                    / pl.col("total_time")
                    / pl.col("population_country")).alias("with_trf_light_norm"),
                ((pl.col("without_trf_light_day_country") + pl.col("without_trf_light_night_country"))
                    / pl.col("total_time")
                    / pl.col("population_country")).alias("without_trf_light_norm"),
                pl.col("country").str.to_titlecase().alias("country")]))
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

        SPEED_COL = "speed_crossing_day_night_country_avg"
        TIME_COL = "time_crossing_day_night_country_avg"

        def _scalar_select(df: pl.DataFrame, expr: pl.Expr):
            """Return the single scalar from df.select(expr), or None if empty."""
            out = df.select(expr)
            if out.height == 0:
                return None
            s = out.to_series(0)
            return s[0] if len(s) else None

        def _row_value(df: pl.DataFrame, row_idx: int, col: str):
            """Get a value by row index; assumes row_idx is valid."""
            return df.row(row_idx, named=True)[col]

        if df_countries.height == 0:
            logger.error("df_countries is empty; cannot compute stats.")
        else:
            # Exclude zero values before finding min
            nonzero_speed = df_countries.filter(pl.col(SPEED_COL) > 0)
            nonzero_time = df_countries.filter(pl.col(TIME_COL) > 0)

            # Row indices (None if not computable)
            max_speed_idx = _scalar_select(df_countries, pl.col(SPEED_COL).arg_max())
            max_time_idx = _scalar_select(df_countries, pl.col(TIME_COL).arg_max())

            min_speed_idx = None
            if nonzero_speed.height > 0:
                min_speed_idx = (
                    nonzero_speed
                    .with_row_index("idx")
                    .sort(SPEED_COL)
                    .select("idx")
                    .to_series(0)[0]
                )

            min_time_idx = None
            if nonzero_time.height > 0:
                min_time_idx = (
                    nonzero_time
                    .with_row_index("idx")
                    .sort(TIME_COL)
                    .select("idx")
                    .to_series(0)[0]
                )

            # Mean and standard deviation (non-zero only)
            speed_mean = _scalar_select(nonzero_speed, pl.col(SPEED_COL).mean())
            speed_std = _scalar_select(nonzero_speed, pl.col(SPEED_COL).std())
            time_mean = _scalar_select(nonzero_time,  pl.col(TIME_COL).mean())
            time_std = _scalar_select(nonzero_time,  pl.col(TIME_COL).std())

            # Logging (guard against None indices)
            if max_speed_idx is not None:
                max_speed_country = _row_value(df_countries, int(max_speed_idx), "country")
                max_speed_value = float(_row_value(df_countries, int(max_speed_idx), SPEED_COL))
                logger.info(
                    f"Country with the highest average speed while crossing: {max_speed_country} "
                    f"({max_speed_value:.2f})"
                )
            else:
                logger.info("No max speed could be computed (empty or invalid speed column).")

            if min_speed_idx is not None:
                min_speed_country = _row_value(df_countries, int(min_speed_idx), "country")
                min_speed_value = float(_row_value(df_countries, int(min_speed_idx), SPEED_COL))
                logger.info(
                    f"Country with the lowest non-zero average speed while crossing: {min_speed_country} "
                    f"({min_speed_value:.2f})"
                )
            else:
                logger.info("No non-zero speed rows found; cannot compute min non-zero speed.")

            logger.info(f"Mean speed while crossing (non-zero): {float(speed_mean) if speed_mean is not None else float('nan'):.2f}")  # noqa: E501
            logger.info(f"Standard deviation of speed while crossing (non-zero): {float(speed_std) if speed_std is not None else float('nan'):.2f}")  # noqa: E501

            if max_time_idx is not None:
                max_time_country = _row_value(df_countries, int(max_time_idx), "country")
                max_time_value = float(_row_value(df_countries, int(max_time_idx), TIME_COL))
                logger.info(
                    f"Country with the highest average crossing time: {max_time_country} "
                    f"({max_time_value:.2f})"
                )
            else:
                logger.info("No max time could be computed (empty or invalid time column).")

            if min_time_idx is not None:
                min_time_country = _row_value(df_countries, int(min_time_idx), "country")
                min_time_value = float(_row_value(df_countries, int(min_time_idx), TIME_COL))
                logger.info(
                    f"Country with the lowest non-zero average crossing time: {min_time_country} "
                    f"({min_time_value:.2f})"
                )
            else:
                logger.info("No non-zero time rows found; cannot compute min non-zero time.")

            logger.info(f"Mean crossing time (non-zero): {float(time_mean) if time_mean is not None else float('nan'):.2f}")  # noqa: E501
            logger.info(f"Standard deviation of crossing time (non-zero): {float(time_std) if time_std is not None else float('nan'):.2f}")  # noqa: E501

            # Equivalent of: df_countries[['total_time','total_videos']].agg(['mean','std','sum'])
            stats = df_countries.select([
                pl.col("total_time").mean().alias("total_time_mean"),
                pl.col("total_time").std().alias("total_time_std"),
                pl.col("total_time").sum().alias("total_time_sum"),
                pl.col("total_videos").mean().alias("total_videos_mean"),
                pl.col("total_videos").std().alias("total_videos_std"),
                pl.col("total_videos").sum().alias("total_videos_sum"),
            ])
            if stats.height > 0:
                st = stats.row(0, named=True)
                logger.info(
                    f"Average total_time: {float(st['total_time_mean']):.2f}, "
                    f"Standard deviation: {float(st['total_time_std']):.2f}, "
                    f"Sum: {float(st['total_time_sum']):.2f}"
                )
                logger.info(
                    f"Average total_videos: {float(st['total_videos_mean']):.2f}, "
                    f"Standard deviation: {float(st['total_videos_std']):.2f}, "
                    f"Sum: {float(st['total_videos_sum']):.2f}"
                )

            # Max/min total_time rows (guarded)
            max_total_time_idx = _scalar_select(df_countries, pl.col("total_time").arg_max())
            if max_total_time_idx is not None:
                max_row = df_countries.row(int(max_total_time_idx), named=True)
                logger.info(
                    f"Country with maximum total_time: {max_row['country']}, "
                    f"total_time: {max_row['total_time']}, "
                    f"total_videos: {max_row['total_videos']}"
                )

            min_total_time_idx = _scalar_select(df_countries, pl.col("total_time").arg_min())
            if min_total_time_idx is not None:
                min_row = df_countries.row(int(min_total_time_idx), named=True)
                logger.info(
                    f"Country with minimum total_time: {min_row['country']}, "
                    f"total_time: {min_row['total_time']}, "
                    f"total_videos: {min_row['total_videos']}"
                )
        logger.info("Analysis complete.")
