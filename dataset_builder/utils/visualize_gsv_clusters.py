import argparse
import os
from typing import Dict, Tuple

import folium
import numpy as np
import pandas as pd
from folium.plugins import MarkerCluster


EARTH_RADIUS_M = 6371000.0


def destination_point(lat_deg: float, lon_deg: float, bearing_deg: float, distance_m: float) -> Tuple[float, float]:
    """
    Compute the destination point from start point, initial bearing, and distance on Earth.
    Returns (lat_deg, lon_deg).
    """
    lat1 = np.radians(lat_deg)
    lon1 = np.radians(lon_deg)
    brng = np.radians(bearing_deg % 360.0)
    ang_dist = distance_m / EARTH_RADIUS_M

    lat2 = np.arcsin(np.sin(lat1) * np.cos(ang_dist) + np.cos(lat1) * np.sin(ang_dist) * np.cos(brng))
    lon2 = lon1 + np.arctan2(
        np.sin(brng) * np.sin(ang_dist) * np.cos(lat1),
        np.cos(ang_dist) - np.sin(lat1) * np.sin(lat2),
    )

    return float(np.degrees(lat2)), float((np.degrees(lon2) + 540) % 360 - 180)


def color_for_cluster(cluster_id: int) -> str:
    """Pick a stable color from a palette for a cluster id."""
    palette = [
        "red",
        "blue",
        "green",
        "purple",
        "orange",
        "darkred",
        "lightred",
        "beige",
        "darkblue",
        "darkgreen",
        "cadetblue",
        "darkpurple",
        "white",
        "pink",
        "lightblue",
        "lightgreen",
        "gray",
        "black",
        "lightgray",
    ]
    return palette[cluster_id % len(palette)]


def build_map(
    centers_csv: str,
    assignments_csv: str = "",
    show_all_images: bool = False,
    output_html: str = "",
) -> str:
    centers_df = pd.read_csv(centers_csv)
    if centers_df.empty:
        raise SystemExit("Centers CSV is empty.")

    if assignments_csv and os.path.exists(assignments_csv):
        assign_df = pd.read_csv(assignments_csv)
    else:
        assign_df = pd.DataFrame()

    center_lat = float(centers_df["center_lat"].mean())
    center_lon = float(centers_df["center_lon"].mean())
    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=13, control_scale=True)

    # Draw centers with circle and heading ray
    for _, row in centers_df.iterrows():
        cid = int(row["cluster_id"])
        lat = float(row["center_lat"])
        lon = float(row["center_lon"])
        heading = float(row["center_heading"]) if "center_heading" in row else 0.0
        radius_m = float(row.get("radius_m", 15.0))
        count = int(row.get("count", 0))
        pano = str(row.get("representative_panoid", ""))
        min_year = int(row.get("min_year", 0))
        max_year = int(row.get("max_year", 0))
        color = color_for_cluster(cid)

        folium.Circle(
            location=[lat, lon],
            radius=radius_m,
            color=color,
            fill=True,
            fill_opacity=0.15,
            weight=2,
            popup=folium.Popup(
                html=(
                    f"<b>Cluster</b>: {cid}<br>"
                    f"<b>Count</b>: {count}<br>"
                    f"<b>Years</b>: {min_year}-{max_year}<br>"
                    f"<b>Rep panoid</b>: {pano}"
                ),
                max_width=300,
            ),
        ).add_to(fmap)

        # Heading ray (short line indicating viewing direction)
        end_lat, end_lon = destination_point(lat, lon, heading, max(10.0, radius_m * 0.9))
        folium.PolyLine(
            locations=[(lat, lon), (end_lat, end_lon)],
            color=color,
            weight=3,
            opacity=0.9,
        ).add_to(fmap)

        # Center point marker on top
        folium.CircleMarker(
            location=[lat, lon], radius=4, color=color, fill=True, fill_opacity=1.0
        ).add_to(fmap)

    # Optionally add all image points colored by cluster
    if show_all_images and not assign_df.empty and "cluster_id" in assign_df.columns:
        cluster_layer = MarkerCluster(name="All images").add_to(fmap)
        for _, r in assign_df.iterrows():
            cid = int(r["cluster_id"]) if not np.isnan(r["cluster_id"]) else -1
            lat = float(r["lat"])
            lon = float(r["lon"])
            year = int(r.get("year", 0))
            month = int(r.get("month", 0))
            head = float(r.get("northdeg", 0.0))
            panoid = str(r.get("panoid", ""))
            folium.CircleMarker(
                location=[lat, lon],
                radius=2,
                color=color_for_cluster(cid),
                fill=True,
                fill_opacity=0.7,
                popup=folium.Popup(
                    html=(
                        f"<b>Cluster</b>: {cid}<br>"
                        f"<b>Date</b>: {year}-{month:02d}<br>"
                        f"<b>Heading</b>: {head:.1f}<br>"
                        f"<b>Panoid</b>: {panoid}"
                    ),
                    max_width=300,
                ),
            ).add_to(cluster_layer)

    if not output_html:
        # Default output path
        city = str(centers_df["city_id"].iloc[0]) if "city_id" in centers_df.columns else "city"
        os.makedirs("visualizations", exist_ok=True)
        output_html = os.path.join("visualizations", f"clusters_{city}.html")

    fmap.save(output_html)
    return output_html


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize clustered GSV city data using Folium.")
    parser.add_argument(
        "--clustered-dir",
        type=str,
        default="Dataframes_clustered",
        help="Directory containing per-city clustered CSVs",
    )
    parser.add_argument(
        "--city",
        type=str,
        required=True,
        help="City to visualize (without .csv)",
    )
    parser.add_argument(
        "--show-all-images",
        action="store_true",
        help="If set, plot every image point colored by cluster",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional explicit output HTML path",
    )

    args = parser.parse_args()
    centers_csv = os.path.join(args.clustered_dir, f"{args.city}_centers.csv")
    assign_csv = os.path.join(args.clustered_dir, f"{args.city}.csv")

    if not os.path.exists(centers_csv):
        raise SystemExit(f"Centers CSV not found: {centers_csv}")

    output_path = build_map(
        centers_csv=centers_csv,
        assignments_csv=assign_csv,
        show_all_images=args.show_all_images,
        output_html=args.output,
    )
    print(f"Saved map to: {output_path}")


if __name__ == "__main__":
    main()


