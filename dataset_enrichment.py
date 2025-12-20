from typing import Optional
import pandas as pd
import common
import world_bank_data as wb
from custom_logger import CustomLogger

logger = CustomLogger(__name__)


class DatasetEnrichment:
    def __init__(self) -> None:
        pass

    def get_iso_alpha_3(self, country_name: str, existing_iso: Optional[str]) -> Optional[str]:
        try:
            import pycountry
            return pycountry.countries.lookup(country_name).alpha_3
        except Exception:
            if country_name.strip().upper() == "KOSOVO":
                return "XKX"
            return existing_iso if existing_iso else None

    def get_latest_population(self) -> pd.DataFrame:
        indicator = "SP.POP.TOTL"
        population_data = wb.get_series(indicator, id_or_value="id", mrv=1)
        df = population_data.reset_index()
        df = df.rename(columns={df.columns[0]: "iso3", df.columns[2]: "Year", df.columns[3]: "Population"})
        df["Population"] = df["Population"] / 1000
        return df

    def update_population_in_csv(self, data: pd.DataFrame) -> None:
        if "iso3" not in data.columns:
            raise KeyError("The CSV file does not have a 'iso3' column.")
        if "population_country" not in data.columns:
            data["population_country"] = None

        latest_population = self.get_latest_population()
        population_dict = dict(zip(latest_population["iso3"], latest_population["Population"]))
        for idx, row in data.iterrows():
            iso3 = row["iso3"]
            data.at[idx, "population_country"] = population_dict.get(iso3, None)  # type: ignore

        mapping_path = common.get_configs("mapping")
        data.to_csv(mapping_path, index=False)
        logger.info("Mapping file updated successfully with country population.")

    def get_latest_gini_values(self) -> pd.DataFrame:
        indicator = "SI.POV.GINI"
        gini_data = wb.get_series(indicator, id_or_value="id", mrv=1)
        df = gini_data.reset_index()
        df = df.rename(columns={df.columns[0]: "iso3", df.columns[2]: "Year", df.columns[3]: "gini"})
        df = df.sort_values(by=["iso3", "Year"], ascending=[True, False]).drop_duplicates(subset=["iso3"])
        return df[["iso3", "gini"]]

    def fill_gini_data(self, df: pd.DataFrame) -> None:
        try:
            if "iso3" not in df.columns:
                logger.error("Missing column 'iso3'.")
                return
            if "gini" not in df.columns:
                df["gini"] = None

            gini_df = self.get_latest_gini_values()
            updated_df = pd.merge(df, gini_df, on="iso3", how="left", suffixes=("", "_new"))
            updated_df["gini"] = updated_df["gini_new"].combine_first(updated_df["gini"])
            updated_df = updated_df.drop(columns=["gini_new"])

            mapping_path = common.get_configs("mapping")
            updated_df.to_csv(mapping_path, index=False)
            logger.info("Mapping file updated successfully with GINI value.")
        except Exception as e:
            logger.error(f"An error occurred: {e}")
