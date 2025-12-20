# by Pavlo Bazilinskyy <pavlo.bazilinskyy@gmail.com>
import pandas as pd
import os
import common
from custom_logger import CustomLogger
from logmod import logs

logs(show_level=common.get_configs("logger_level"), show_color=True)
logger = CustomLogger(__name__)  # use custom logger

# average age data from https://simplemaps.com/data/countries
age_data = pd.read_csv(os.path.join(common.root_dir, 'countries.csv'))


# Function to fetch median age for a given ISO3 code
def get_country_median_age(iso2_code):
    try:
        # Filter the dataset by country name
        row = age_data[age_data['iso2'].str.lower() == iso2_code.lower()]
        if not row.empty:
            # Return median age
            return row.iloc[0]['median_age']
        else:
            return 0.0
    except Exception as e:
        print(f"Error fetching height data: {e}")
        return 0.0


if __name__ == "__main__":
    # Load the CSV
    df = pd.read_csv(common.get_configs("mapping"))
    # Update the med_age column only if it's 0.0 or missing
    for idx, row in df.iterrows():
        if pd.isnull(row['med_age']) or row['med_age'] == 0.0:
            iso3 = row['iso3']
            iso2 = common.get_iso2_country_code(row['country'])
            if pd.notnull(iso3):
                median_age = get_country_median_age(iso2)
                if median_age is not None:
                    df.at[idx, 'med_age'] = median_age  # type: ignore
                    logger.info(f"✅ Fetched age for {iso3}: {median_age}")
                else:
                    logger.warning(f"⚠️ No data for {iso3}")
    # Save the updated DataFrame to a new CSV file
    output_file = 'mapping_with_med_age.csv'
    df.to_csv(output_file, index=False)
    logger.info(f"✅ Updated average age values saved to {output_file}")
