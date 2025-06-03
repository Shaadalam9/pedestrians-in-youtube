import pandas as pd
import requests
import common
from custom_logger import CustomLogger
from logmod import logs

logs(show_level=common.get_configs("logger_level"), show_color=True)
logger = CustomLogger(__name__)  # use custom logger

# WHO GHO OData API base URL
api_base_url = "https://ghoapi.azureedge.net/api/"

# Indicator code for population median age
indicator_code = "WHOSIS_000001"


# Function to fetch median age for a given ISO3 code
def fetch_median_age(iso3):
    try:
        url = f"{api_base_url}{indicator_code}/Country/{iso3}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if 'value' in data and data['value']:
                # Assuming the latest value is the first in the list
                return data['value'][0]['Value']
        return None
    except Exception as e:
        logger.error(f"Error fetching data for {iso3}: {e}")
        return None


if __name__ == "__main__":
    # Load the CSV
    df = pd.read_csv(common.get_configs("mapping"))
    # Update the avg_age column only if it's 0.0 or missing
    for idx, row in df.iterrows():
        if pd.isnull(row['avg_age']) or row['avg_age'] == 0.0:
            iso3 = row['iso3']
            if pd.notnull(iso3):
                median_age = fetch_median_age(iso3)
                if median_age is not None:
                    df.at[idx, 'avg_age'] = median_age
                    logger.info(f"✅ Fetched age for {iso3}: {median_age}")
                else:
                    logger.warning(f"⚠️ No data for {iso3}")
    # Save the updated DataFrame to a new CSV file
    output_file = 'mapping_with_avg_age.csv'
    df.to_csv(output_file, index=False)
    logger.info(f"✅ Updated average age values saved to {output_file}")