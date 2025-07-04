"""
Working with the Kymata Atlas API.
"""
import json
from typing import Collection

import requests


API_URL = "https://kymata.org/api/functions/"


def fetch_data_dict(api: str = API_URL) -> dict:
    """
    Fetches data from Kymata API and returns it as a dictionary.

    Params
    ------
        api : URL of the API from which to fetch data

    Returns
    -------
        API response object in a dictionary form

    Raises:
        ConnectionError: If there's an issue fetching data from the API.
    """
    try:
        response = requests.get(api)
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)
        return json.loads(response.text)
    except requests.exceptions.RequestException as e:
        raise ConnectionError(f"Failed to fetch data from {api}: {e}")


def verify_kids(kids: Collection[str]) -> None:
    """
    Verifies a list of KIDs against the API.

    Returns silently if all KIDs validate.

    Params
    ------
    kids (Collection[str]): KIDs to verify.

    Raises:
        ConnectionError if the API is unavailable.
        ValueError if one or more of the KIDs is invalid.
    """
    api_data = fetch_data_dict(API_URL)
    valid_api_kids = {
        item["kid"]
        for item in api_data
        if "kid" in item
    }
    # Check if all KIDs in transform_KIDs exist in valid_api_kids
    missing_api_kids = set(kids) - valid_api_kids
    if missing_api_kids:
        raise ValueError(f"The following KIDs from transform_KIDs do not exist in the API:"
                         f" {sorted(list(missing_api_kids))}")
