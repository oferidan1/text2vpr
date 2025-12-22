import argparse
import math
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


EARTH_RADIUS_M = 6371000.0


def haversine_distance_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Compute great-circle distance between two points on Earth (in meters).
    Inputs/outputs in degrees for lat/lon, meters for distance.
    """
    rlat1 = math.radians(lat1)
    rlat2 = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2.0) ** 2 + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return EARTH_RADIUS_M * c


def circular_angle_diff_deg(a: float, b: float) -> float:
    """Smallest absolute difference between two headings (degrees 0..360)."""
    diff = abs((a - b) % 360.0)
    if diff > 180.0:
        diff = 360.0 - diff
    return diff


def circular_mean_deg(angles_deg: np.ndarray) -> float:
    """
    Circular mean of headings in degrees (0..360).
    Returns value in [0, 360).
    """
    radians = np.deg2rad(angles_deg)
    mean_sin = np.mean(np.sin(radians))
    mean_cos = np.mean(np.cos(radians))
    if mean_sin == 0.0 and mean_cos == 0.0:
        return 0.0
    mean_angle = math.degrees(math.atan2(mean_sin, mean_cos))
    if mean_angle < 0.0:
        mean_angle += 360.0
    return mean_angle


def approximate_degree_window(lat_deg: float, radius_m: float) -> Tuple[float, float]:
    """
    Return (dlat_deg, dlon_deg) window approximations for a radius in meters at given latitude.
    Good for candidate filtering; final filtering should still use haversine.
    """
    dlat = radius_m / 111320.0
    cos_lat = max(1e-6, math.cos(math.radians(lat_deg)))
    dlon = radius_m / (111320.0 * cos_lat)
    return dlat, dlon


def select_representative_index(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    headings_deg: np.ndarray,
) -> int:
    """
    Pick a representative (medoid-style) index for a cluster: the point with minimal
    sum of spatial rank + heading deviation from circular mean. Returns index within cluster arrays.
    """
    center_lat = float(np.mean(latitudes))
    center_lon = float(np.mean(longitudes))
    mean_heading = circular_mean_deg(headings_deg)

    # Spatial distances to center
    spatial = np.array(
        [
            haversine_distance_meters(center_lat, center_lon, float(latitudes[i]), float(longitudes[i]))
            for i in range(latitudes.shape[0])
        ]
    )
    spatial_rank = np.argsort(np.argsort(spatial))  # rank of spatial proximity

    heading_dev = np.array([circular_angle_diff_deg(float(h), mean_heading) for h in headings_deg])
    heading_rank = np.argsort(np.argsort(heading_dev))

    score = spatial_rank + heading_rank
    return int(np.argmin(score))


def greedy_cluster_city(
    city_df: pd.DataFrame,
    radius_m: float,
    heading_tolerance_deg: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Greedy clustering by spatial radius and heading tolerance.

    - Start from unassigned images; pick current point as a seed.
    - Candidate window via degree approximation; refine using haversine.
    - Within spatial neighbors, keep those with heading within tolerance of the seed heading.
    - Form cluster, compute center coordinates (mean) and center heading (circular mean).

    Returns:
    - assignments_df: original df with a new 'cluster_id' column (int)
    - centers_df: one row per cluster with center metadata
    """
    required_columns = ["place_id", "year", "month", "northdeg", "city_id", "lat", "lon", "panoid"]
    missing = [c for c in required_columns if c not in city_df.columns]
    if missing:
        raise ValueError(f"Missing required columns in city dataframe: {missing}")

    df = city_df.reset_index(drop=True).copy()
    num_rows = df.shape[0]
    assigned = np.zeros(num_rows, dtype=bool)
    cluster_ids = np.full(num_rows, -1, dtype=int)

    lats = df["lat"].to_numpy(dtype=float)
    lons = df["lon"].to_numpy(dtype=float)
    heads = df["northdeg"].to_numpy(dtype=float)

    cluster_centers: List[Dict] = []
    cluster_index = 0

    # Process in a spatially coherent order to improve cache locality
    order = np.lexsort((lons, lats))

    for idx in order:
        if assigned[idx]:
            continue

        seed_lat = lats[idx]
        seed_lon = lons[idx]
        seed_head = heads[idx]

        dlat_deg, dlon_deg = approximate_degree_window(seed_lat, radius_m)
        lat_min, lat_max = seed_lat - dlat_deg, seed_lat + dlat_deg
        lon_min, lon_max = seed_lon - dlon_deg, seed_lon + dlon_deg

        # Candidate mask by bounding box first
        window_mask = (
            (lats >= lat_min)
            & (lats <= lat_max)
            & (lons >= lon_min)
            & (lons <= lon_max)
            & (~assigned)
        )
        candidate_indices = np.where(window_mask)[0]

        if candidate_indices.size == 0:
            # Fallback: seed becomes its own cluster
            cluster_ids[idx] = cluster_index
            assigned[idx] = True
            cluster_centers.append(
                {
                    "cluster_id": cluster_index,
                    "city_id": df.at[idx, "city_id"],
                    "center_lat": seed_lat,
                    "center_lon": seed_lon,
                    "center_heading": seed_head % 360.0,
                    "count": 1,
                    "representative_panoid": df.at[idx, "panoid"],
                    "min_year": int(df.at[idx, "year"]),
                    "max_year": int(df.at[idx, "year"]),
                    "radius_m": float(radius_m),
                    "heading_tolerance_deg": float(heading_tolerance_deg),
                }
            )
            cluster_index += 1
            continue

        # Refine by true haversine distance and heading tolerance
        spatial_ok: List[int] = []
        for j in candidate_indices:
            d = haversine_distance_meters(seed_lat, seed_lon, lats[j], lons[j])
            if d <= radius_m:
                spatial_ok.append(j)

        if not spatial_ok:
            # No spatial neighbors; seed alone
            cluster_ids[idx] = cluster_index
            assigned[idx] = True
            cluster_centers.append(
                {
                    "cluster_id": cluster_index,
                    "city_id": df.at[idx, "city_id"],
                    "center_lat": seed_lat,
                    "center_lon": seed_lon,
                    "center_heading": seed_head % 360.0,
                    "count": 1,
                    "representative_panoid": df.at[idx, "panoid"],
                    "min_year": int(df.at[idx, "year"]),
                    "max_year": int(df.at[idx, "year"]),
                    "radius_m": float(radius_m),
                    "heading_tolerance_deg": float(heading_tolerance_deg),
                }
            )
            cluster_index += 1
            continue

        same_heading_members: List[int] = []
        for j in spatial_ok:
            if circular_angle_diff_deg(heads[j], seed_head) <= heading_tolerance_deg:
                same_heading_members.append(j)

        if not same_heading_members:
            # Seed alone (heading tolerance filtered all out)
            cluster_ids[idx] = cluster_index
            assigned[idx] = True
            cluster_centers.append(
                {
                    "cluster_id": cluster_index,
                    "city_id": df.at[idx, "city_id"],
                    "center_lat": seed_lat,
                    "center_lon": seed_lon,
                    "center_heading": seed_head % 360.0,
                    "count": 1,
                    "representative_panoid": df.at[idx, "panoid"],
                    "min_year": int(df.at[idx, "year"]),
                    "max_year": int(df.at[idx, "year"]),
                    "radius_m": float(radius_m),
                    "heading_tolerance_deg": float(heading_tolerance_deg),
                }
            )
            cluster_index += 1
            continue

        member_indices = np.array(sorted(set(same_heading_members)), dtype=int)

        # Compute cluster center and representative
        cluster_lats = lats[member_indices]
        cluster_lons = lons[member_indices]
        cluster_heads = heads[member_indices]
        center_lat = float(np.mean(cluster_lats))
        center_lon = float(np.mean(cluster_lons))
        center_heading = circular_mean_deg(cluster_heads)

        rep_local_idx = select_representative_index(cluster_lats, cluster_lons, cluster_heads)
        rep_global_idx = int(member_indices[rep_local_idx])

        # Assign
        cluster_ids[member_indices] = cluster_index
        assigned[member_indices] = True

        years = df.loc[member_indices, "year"].astype(int)
        cluster_centers.append(
            {
                "cluster_id": cluster_index,
                "city_id": df.at[idx, "city_id"],
                "center_lat": center_lat,
                "center_lon": center_lon,
                "center_heading": center_heading,
                "count": int(member_indices.size),
                "representative_panoid": df.at[rep_global_idx, "panoid"],
                "min_year": int(years.min()),
                "max_year": int(years.max()),
                "radius_m": float(radius_m),
                "heading_tolerance_deg": float(heading_tolerance_deg),
            }
        )

        cluster_index += 1

    assignments_df = df.copy()
    assignments_df["cluster_id"] = cluster_ids

    centers_df = pd.DataFrame(cluster_centers)
    centers_df = centers_df.sort_values(["city_id", "cluster_id"]).reset_index(drop=True)

    return assignments_df, centers_df


def process_city(
    city_csv_path: str,
    output_dir: str,
    radius_m: float,
    heading_tolerance_deg: float,
) -> Tuple[str, str]:
    city_df = pd.read_csv(city_csv_path)
    if "city_id" not in city_df.columns:
        # Infer city name from file name if needed
        city_name = os.path.splitext(os.path.basename(city_csv_path))[0]
        city_df = city_df.copy()
        city_df["city_id"] = city_name

    assignments_df, centers_df = greedy_cluster_city(city_df, radius_m, heading_tolerance_deg)

    os.makedirs(output_dir, exist_ok=True)
    city_name = str(assignments_df["city_id"].iloc[0])
    out_assign = os.path.join(output_dir, f"{city_name}.csv")
    out_centers = os.path.join(output_dir, f"{city_name}_centers.csv")
    assignments_df.to_csv(out_assign, index=False)
    centers_df.to_csv(out_centers, index=False)
    return out_assign, out_centers


def list_city_csvs(dataframes_dir: str, only_city: str = "") -> List[str]:
    all_csvs = []
    for name in sorted(os.listdir(dataframes_dir)):
        if not name.lower().endswith(".csv"):
            continue
        if only_city and os.path.splitext(name)[0].lower() != only_city.lower():
            continue
        all_csvs.append(os.path.join(dataframes_dir, name))
    return all_csvs


def main() -> None:
    parser = argparse.ArgumentParser(description="Cluster GSV city data by radius and heading.")
    parser.add_argument(
        "--dataframes-dir",
        type=str,
        default="Dataframes",
        help="Directory containing per-city CSVs (e.g., Paris.csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="Dataframes_clustered",
        help="Directory to write clustered outputs",
    )
    parser.add_argument(
        "--city",
        type=str,
        default="",
        help="Optional single city to process (without .csv). If omitted, process all.",
    )
    parser.add_argument(
        "--radius-m",
        type=float,
        default=15.0,
        help="Spatial radius in meters for clustering.",
    )
    parser.add_argument(
        "--heading-tolerance-deg",
        type=float,
        default=20.0,
        help="Maximum heading difference (degrees) to be considered same viewpoint.",
    )

    args = parser.parse_args()

    city_paths = list_city_csvs(args.dataframes_dir, args.city)
    if not city_paths:
        raise SystemExit("No city CSVs found to process.")

    print(
        f"Clustering {len(city_paths)} city CSV(s) with radius={args.radius_m} m, "
        f"heading_tol={args.heading_tolerance_deg} deg"
    )

    for csv_path in city_paths:
        city_name = os.path.splitext(os.path.basename(csv_path))[0]
        print(f"- Processing {city_name}...")
        out_assign, out_centers = process_city(
            csv_path, args.output_dir, args.radius_m, args.heading_tolerance_deg
        )
        print(f"  wrote: {out_assign}")
        print(f"  wrote: {out_centers}")


if __name__ == "__main__":
    main()


