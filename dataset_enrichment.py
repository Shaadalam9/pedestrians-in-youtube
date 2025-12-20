"""Dataset enrichment utilities.

This module provides helper methods to enrich a mapping dataset with:
  - ISO alpha-3 country codes
  - Latest available population (World Bank: SP.POP.TOTL), scaled to thousands
  - Latest available GINI index (World Bank: SI.POV.GINI)

Data sources:
  - `world_bank_data` is used to fetch indicator series with "most recent value" (mrv=1).
  - `pycountry` is used to resolve country names to ISO alpha-3 codes where possible.

Operational notes:
  - Enrichment methods are designed to be tolerant of missing values and partial coverage.
  - CSV writes are performed to the configured mapping path (`common.get_configs("mapping")`).
"""

from typing import Optional
import pandas as pd
import common
import world_bank_data as wb
from custom_logger import CustomLogger

# Module-level logger to ensure consistent logging across enrichment operations.
logger = CustomLogger(__name__)


class DatasetEnrichment:
    """Enrich a mapping DataFrame with country-level attributes.

    This class centralizes enrichment steps that typically operate on a mapping CSV,
    adding or updating columns in-place and persisting results to disk.

    Notes:
        - Methods that "update" write to the mapping CSV path from configuration.
        - World Bank indicators may not be available for all ISO3 codes; missing
          values are preserved as None.
    """

    def __init__(self) -> None:
        """Initialize the dataset enrichment helper."""
        # Stateless helper; initialization kept for future extension.
        pass

    def get_iso_alpha_3(self, country_name: str, existing_iso: Optional[str]) -> Optional[str]:
        """Resolve a country name to an ISO alpha-3 code.

        Args:
            country_name: Human-readable country name (as found in the dataset).
            existing_iso: Existing ISO3 value (if already present) used as fallback.

        Returns:
            ISO alpha-3 code when resolvable, otherwise:
              - "XKX" for Kosovo (special-case handling),
              - existing_iso if provided,
              - None if no value can be determined.

        Notes:
            `pycountry` performs best-effort name lookup and may raise exceptions for
            unknown or ambiguous names; this method intentionally catches failures.
        """
        try:
            import pycountry
            return pycountry.countries.lookup(country_name).alpha_3
        except Exception:
            # Kosovo is commonly represented as XKX in many datasets despite not
            # being part of the ISO 3166-1 official list in some contexts.
            if country_name.strip().upper() == "KOSOVO":
                return "XKX"
            # Preserve existing value if one exists; otherwise return None.
            return existing_iso if existing_iso else None

    def get_latest_population(self) -> pd.DataFrame:
        """Fetch the latest available population per country from World Bank.

        Uses indicator SP.POP.TOTL and requests the most recent value (mrv=1).
        The returned population is scaled to thousands (Population / 1000).

        Returns:
            DataFrame with columns:
              - iso3: ISO alpha-3 country code
              - Year: year of the reported value (as returned by the API)
              - Population: population in thousands
        """
        indicator = "SP.POP.TOTL"
        population_data = wb.get_series(indicator, id_or_value="id", mrv=1)
        df = population_data.reset_index()
        # The source series uses index columns; normalize into stable column names.
        df = df.rename(columns={df.columns[0]: "iso3", df.columns[2]: "Year", df.columns[3]: "Population"})
        # Store in thousands to reduce magnitude and match typical reporting conventions.
        df["Population"] = df["Population"] / 1000
        return df

    def update_population_in_csv(self, data: pd.DataFrame) -> None:
        """Update (or add) `population_country` based on latest World Bank population.

        This method:
          - validates that the DataFrame includes an `iso3` column,
          - ensures `population_country` exists,
          - maps ISO3 -> population (thousands) and writes into the DataFrame,
          - persists the updated DataFrame to the configured mapping CSV path.

        Args:
            data: Mapping DataFrame to update in-place.

        Returns:
            None.

        Raises:
            KeyError: If the input DataFrame does not include an `iso3` column.
        """
        # ISO3 is required to join World Bank data; fail fast if missing.
        if "iso3" not in data.columns:
            raise KeyError("The CSV file does not have a 'iso3' column.")
        # Create the target column if not already present.
        if "population_country" not in data.columns:
            data["population_country"] = None

        latest_population = self.get_latest_population()
        population_dict = dict(zip(latest_population["iso3"], latest_population["Population"]))

        # Iterate row-wise to preserve existing row order and avoid introducing merges.
        for idx, row in data.iterrows():
            iso3 = row["iso3"]
            data.at[idx, "population_country"] = population_dict.get(iso3, None)  # type: ignore

        # Persist results to the configured mapping location.
        mapping_path = common.get_configs("mapping")
        data.to_csv(mapping_path, index=False)
        logger.info("Mapping file updated successfully with country population.")

    def get_latest_gini_values(self) -> pd.DataFrame:
        """Fetch the latest available GINI index per country from World Bank.

        Uses indicator SI.POV.GINI and requests the most recent value (mrv=1).
        Some countries may have sparse reporting; the method sorts by year and
        keeps the newest record per ISO3.

        Returns:
            DataFrame with columns:
              - iso3: ISO alpha-3 country code
              - gini: latest available GINI value
        """
        indicator = "SI.POV.GINI"
        gini_data = wb.get_series(indicator, id_or_value="id", mrv=1)
        df = gini_data.reset_index()
        # Normalize column names to stable identifiers used downstream.
        df = df.rename(columns={df.columns[0]: "iso3", df.columns[2]: "Year", df.columns[3]: "gini"})
        # Ensure one row per country (most recent year retained).
        df = df.sort_values(by=["iso3", "Year"], ascending=[True, False]).drop_duplicates(subset=["iso3"])
        return df[["iso3", "gini"]]

    def fill_gini_data(self, df: pd.DataFrame) -> None:
        """Fill missing/empty GINI values in a mapping dataset and persist to CSV.

        This method:
          - validates required `iso3` column existence,
          - ensures a `gini` column exists,
          - merges latest World Bank GINI values into the dataset,
          - prefers newly fetched values where available (combine_first),
          - writes the resulting DataFrame to the configured mapping CSV path.

        Args:
            df: Mapping DataFrame to enrich.

        Returns:
            None. The updated mapping is written to disk.

        Notes:
            - If `iso3` is missing, the method logs an error and returns without writing.
            - Exceptions are caught and logged to keep the pipeline resilient.
        """
        try:
            # ISO3 is mandatory for the join; log and exit rather than raising.
            if "iso3" not in df.columns:
                logger.error("Missing column 'iso3'.")
                return
            # Ensure target column exists to support combine_first semantics.
            if "gini" not in df.columns:
                df["gini"] = None

            gini_df = self.get_latest_gini_values()

            # Left join to preserve original rows; suffixes avoid clobbering existing column.
            updated_df = pd.merge(df, gini_df, on="iso3", how="left", suffixes=("", "_new"))
            # Prefer newly retrieved values where present, otherwise keep existing.
            updated_df["gini"] = updated_df["gini_new"].combine_first(updated_df["gini"])
            updated_df = updated_df.drop(columns=["gini_new"])

            mapping_path = common.get_configs("mapping")
            updated_df.to_csv(mapping_path, index=False)
            logger.info("Mapping file updated successfully with GINI value.")
        except Exception as e:
            # Defensive: enrichment should not crash the pipeline.
            logger.error(f"An error occurred: {e}")
