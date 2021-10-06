import argparse
import json
from pprint import pprint
from typing import Dict

import requests

NOMINATIM_API_URL = "https://nominatim.openstreetmap.org"
NOMINATIM_DETAILS_ENDPOINT = f"{NOMINATIM_API_URL}/details"
NOMINATIM_SEARCH_ENDPOINT = f"{NOMINATIM_API_URL}/search"
NOMINATIM_REVERSE_ENDPOINT = f"{NOMINATIM_API_URL}/reverse"

query_params = {
    "namedetails": 1,
    "polygon_geojson": 1,
    "hierarchy": 1,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", type=str, choices=["details", "reverse", "search"])
    return parser.parse_args()


def fetch_osm_details(osm_id: str, osm_type: str, params: Dict[str, int]) -> dict:
    params_query = "&".join(f"{param_name}={param_value}" for param_name, param_value in params.items())
    request_url = f"{NOMINATIM_DETAILS_ENDPOINT}?osmtype={osm_type}&osmid={osm_id}&{params_query}&format=json"
    print(request_url)

    response = requests.get(request_url)
    response.raise_for_status()
    return response.json()


def fetch_osm_search(query: str, params: Dict[str, int]) -> dict:
    params_query = "&".join(f"{param_name}={param_value}" for param_name, param_value in params.items())
    request_url = f"{NOMINATIM_SEARCH_ENDPOINT}?q={query}&{params_query}&format=json"
    print(request_url)

    response = requests.get(request_url)
    response.raise_for_status()
    return response.json()


def fetch_osm_reverse(lat: float, lon: float, zoom: int, params: Dict[str, int]) -> dict:
    params_query = "&".join(f"{param_name}={param_value}" for param_name, param_value in params.items())
    request_url = f"{NOMINATIM_REVERSE_ENDPOINT}?lat={lat}&lon={lon}&zoom={zoom}&{params_query}&format=json"
    print(request_url)

    response = requests.get(request_url)
    response.raise_for_status()
    return response.json()


def main(command):
    if command == "details":
        result = fetch_osm_details(osm_id="175905", osm_type="R", params=query_params)
    elif command == "search":
        result = fetch_osm_search(query="Opole", params=query_params)
    elif command == "reverse":
        result = fetch_osm_reverse(lat=40.7127281, lon=-74.0060152, zoom=10, params=query_params)
    else:
        raise Exception("Wrong command.")

    pprint(result)
    with open(f"{command}_Opole.json", "w") as output_file:
        output_file.write(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    args = parse_args()
    main(args.command)