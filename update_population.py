import pandas as pd
import requests
import common
import pycountry
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import urllib3


# Suppress InsecureRequestWarning
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# Load the CSV file
csv_file = 'mapping.csv'
data = pd.read_csv(csv_file)

# Create a session with retries for better error handling
session = requests.Session()
retries = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
session.mount('https://', HTTPAdapter(max_retries=retries))


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


# Fetch ISO-2 country data
def get_iso2_country_code(country_name):
    try:
        country = pycountry.countries.get(name=country_name)
        if country:
            if country == 'Kosovo':
                return 'XK'
            else:
                return country.alpha_2  # ISO-2 code
        else:
            return "Country not found"
    except KeyError:
        return "Country not found"


# Fetch ISO-3 country data
def get_iso3_country_code(country_name):
    try:
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


# Fetch population by ISO-3 code
def get_country_population(country_data):
    if country_data:
        return country_data[0]['population']
    else:
        return 0.0


# Fetch country data based on its ISO-3 code
def get_country_data(iso3_code):
    try:
        # REST Countries API URL
        api_url = f"https://restcountries.com/v3.1/alpha/{iso3_code}"
        response = session.get(api_url, verify=False)  # Disable SSL verification
        response.raise_for_status()
        if response.status_code == 200:
            country_data = response.json()
            # Extract population from the API response
            return country_data
        else:
            return None
    except Exception as e:
        print(f"Error fetching country data: {e}")
        return None


def fetch_population_data(city, iso2_code, iso3_code):
    city_data = get_city_data(city, iso2_code)
    country_data = get_country_data(iso3_code)
    city_population = get_city_population(city_data)
    country_population = get_country_population(country_data)
    return city_population, country_population


# Iterate over rows and update population columns
for index, row in data.iterrows():
    city = row['city']
    country = row['country']
    city_population, country_population = fetch_population_data(city,
                                                                iso2_code=get_iso2_country_code(country),
                                                                iso3_code=get_iso3_country_code(country))
    print(city, country, city_population, country_population)
    if city_population is not None:
        data.at[index, 'population_city'] = city_population
    if country_population is not None:
        data.at[index, 'population_country'] = country_population

# Save updated CSV
updated_csv_file = 'mapping_updated.csv'
data.to_csv(updated_csv_file, index=False)
print(f"Updated data saved to {updated_csv_file}")
