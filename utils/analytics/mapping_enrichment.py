import common
from tqdm import tqdm
import polars as pl
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
                label="city",
                avg_dict=avg_speed_city,
                value_type="speed",
                col_prefix="speed_crossing",
            ))
        if avg_time_city is not None:
            configs.append(dict(
                label="city",
                avg_dict=avg_time_city,
                value_type="time",
                col_prefix="time_crossing",
            ))
        if avg_speed_country is not None:
            configs.append(dict(
                label="country",
                avg_dict=avg_speed_country,
                value_type="speed",
                col_prefix="speed_crossing",
            ))
        if avg_time_country is not None:
            configs.append(dict(
                label="country",
                avg_dict=avg_time_country,
                value_type="time",
                col_prefix="time_crossing",
            ))

        out = df_mapping

        for cfg in configs:
            label = cfg["label"]
            avg_dict = cfg["avg_dict"]
            col_prefix = cfg["col_prefix"]

            day_col = f"{col_prefix}_day_{label}"
            night_col = f"{col_prefix}_night_{label}"
            avg_col = f"{col_prefix}_day_night_{label}_avg"

            # Ensure columns exist
            for col in [day_col, night_col, avg_col]:
                if col not in out.columns:
                    out = out.with_columns(pl.lit(None).cast(pl.Float64).alias(col))

            # Build an "updates" table from avg_dict
            updates_rows = []

            iterator = avg_dict.items()
            iterator = tqdm(
                iterator,
                desc=f"{label.capitalize()} {cfg['value_type'].capitalize()}s",
                total=len(avg_dict),
            )

            if label == "city":
                # key format: "{city}_{lat}_{long}_{condition}"
                for key, value in iterator:
                    try:
                        parts = key.split("_")
                        city = parts[0]
                        lat = float(parts[1])
                        time_of_day = int(parts[3])
                    except Exception:
                        continue

                    # state lookup, same semantics as pandas version
                    state = metadata_class.get_value(out, "city", city, "lat", lat, "state")

                    updates_rows.append({
                        "city": city,
                        "lat": float(lat),
                        "_state_lookup": state,
                        "_tod": int(time_of_day),
                        "_val": float(value),
                    })

                if not updates_rows:
                    continue

                updates = pl.DataFrame(updates_rows)

                # Decide whether it goes to day or night column
                updates = updates.with_columns([
                    pl.when(pl.col("_tod") == 0).then(pl.col("_val")).otherwise(None).alias("_day"),
                    pl.when(pl.col("_tod") == 1).then(pl.col("_val")).otherwise(None).alias("_night"),
                ])

                # Aggregate in case multiple entries per city/lat/state (take last non-null)
                updates = (
                    updates
                    .group_by(["city", "lat", "_state_lookup"])
                    .agg([
                        pl.col("_day").drop_nulls().last().alias(day_col),
                        pl.col("_night").drop_nulls().last().alias(night_col),
                    ])
                )

                # Join onto mapping. Replicate pandas mask logic:
                # (state matches) OR (both missing)
                #
                # In mapping you may have null/NA; we treat:
                # - mapping_state_str is a string; null stays null
                # - update_state_str is a string; None stays null
                out = out.with_columns([
                    pl.col("state").cast(pl.Utf8, strict=False).alias("_state_str"),
                ])
                updates = updates.with_columns([
                    pl.col("_state_lookup").cast(pl.Utf8, strict=False).alias("_state_str_upd"),
                ])

                # Join by city+lat first (as the strong keys), then conditionally apply based on state match.
                out = out.join(
                    updates.select(["city", "lat", "_state_str_upd", day_col, night_col]),
                    on=["city", "lat"],
                    how="left",
                )

                # Apply state-match semantics to day/night updates (otherwise keep existing)
                state_match = (
                    (pl.col("_state_str") == pl.col("_state_str_upd"))
                    | (pl.col("_state_str").is_null() & pl.col("_state_str_upd").is_null())
                )

                out = out.with_columns([
                    pl.when(state_match & pl.col(day_col).is_not_null())
                      .then(pl.col(day_col))
                      .otherwise(pl.col(day_col).fill_null(pl.col(day_col)))
                      .alias(day_col),
                    pl.when(state_match & pl.col(night_col).is_not_null())
                      .then(pl.col(night_col))
                      .otherwise(pl.col(night_col).fill_null(pl.col(night_col)))
                      .alias(night_col),
                ])

                # Cleanup the joined helper column
                out = out.drop(["_state_str", "_state_str_upd"])

            else:
                # country keys: "{country}_{condition}"
                for key, value in iterator:
                    try:
                        country, tod_s = key.rsplit("_", 1)
                        time_of_day = int(tod_s)
                    except Exception:
                        continue

                    cond_key = f"{country}_{time_of_day}"
                    cross_value = pedestrian_cross_country.get(cond_key, 0)
                    if cross_value <= threshold:
                        continue

                    updates_rows.append({
                        "country": country,
                        "_tod": int(time_of_day),
                        "_val": float(value),
                    })

                if not updates_rows:
                    continue

                updates = pl.DataFrame(updates_rows).with_columns([
                    pl.when(pl.col("_tod") == 0).then(pl.col("_val")).otherwise(None).alias(day_col),
                    pl.when(pl.col("_tod") == 1).then(pl.col("_val")).otherwise(None).alias(night_col),
                ])

                updates = (
                    updates
                    .group_by(["country"])
                    .agg([
                        pl.col(day_col).drop_nulls().last().alias(day_col),
                        pl.col(night_col).drop_nulls().last().alias(night_col),
                    ])
                )

                out = out.join(updates, on="country", how="left", suffix="_upd")

                # Merge joined values into existing columns (prefer new non-null)
                out = out.with_columns([
                    pl.coalesce([pl.col(f"{day_col}_upd"), pl.col(day_col)]).alias(day_col),
                    pl.coalesce([pl.col(f"{night_col}_upd"), pl.col(night_col)]).alias(night_col),
                ]).drop([f"{day_col}_upd", f"{night_col}_upd"])

            # Compute avg_col with same semantics as numpy.where chain:
            # if day>0 and night>0 => mean
            # else if day>0 => day
            # else if night>0 => night
            # else null
            out = out.with_columns(
                pl.when((pl.col(day_col) > 0) & (pl.col(night_col) > 0))
                  .then((pl.col(day_col) + pl.col(night_col)) / 2)
                  .when(pl.col(day_col) > 0)
                  .then(pl.col(day_col))
                  .when(pl.col(night_col) > 0)
                  .then(pl.col(night_col))
                  .otherwise(None)
                  .alias(avg_col)
            )

        return out
