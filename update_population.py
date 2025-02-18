import pandas as pd
import requests
import common
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import urllib3
import time


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
    Get economic or city-related data from the Geonames API, handling pagination.

    :param city_name: Name of the city
    :param country_code: 2-letter ISO code of the country
    :return: Complete city data (list of results)
    """
    base_url = "http://api.geonames.org/searchJSON"
    username = common.get_secrets("geonames_username")  # API username
    results = []
    start_row = 0
    max_rows = 100  # Fetch 100 results per request (max allowed)

    while True:
        params = {
            "q": city_name,
            "country": country_code,
            "username": username,
            "maxRows": max_rows,
            "startRow": start_row
        }

        response = requests.get(base_url, params=params)

        if response.status_code != 200:
            print(f"Error: {response.status_code} - {response.text}")
            return None

        data = response.json()

        if "geonames" not in data or not data["geonames"]:
            break  # No more results

        results.extend(data["geonames"])  # Append new data

        if len(data["geonames"]) < max_rows:
            break  # No more pages

        start_row += max_rows  # Fetch the next page

        time.sleep(1)  # Avoid hitting rate limits

    return results


def get_city_population(city_data):
    """
    Get economic or city-related data from the Geonames API

    :param city_data: city data object.
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


def get_population_data(city, iso2_code, iso3_code):
    city_data = get_city_data(city, iso2_code)
    country_data = get_country_data(iso3_code)
    # Special cases for cities
    if city == 'Boryspil':
        city_population = 64117
    elif city == 'Ibiza':
        city_population = 48684
    elif city == 'Luxembourg City':
        city_population = 136208
    else:
        city_population = get_city_population(city_data)
    # Special cases for countries
    if iso2_code == 'XK':
        country_population = 1578000
    else:
        country_population = get_country_population(country_data)
    return city_population, country_population


if __name__ == "__main__":
    # Iterate over rows and update population columns
    for index, row in data.iterrows():
        city = row['city']
        country = common.correct_country(row['country'])
        iso2_code = common.get_iso2_country_code(country)
        iso3_code = common.get_iso3_country_code(country)
        city_population, country_population = get_population_data(city, iso2_code, iso3_code)
        print(city, country, iso2_code, iso3_code, city_population, country_population)
        if city_population is not None:
            data.at[index, 'population_city'] = city_population
        if country_population is not None:
            data.at[index, 'population_country'] = country_population

    # Save updated CSV
    updated_csv_file = 'mapping_updated.csv'
    data.to_csv(updated_csv_file, index=False)
    print(f"Updated data saved to {updated_csv_file}")
