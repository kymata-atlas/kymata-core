"""
Working with the Kymata Atlas API.
"""
import json

import requests


def fetch_data_dict(api: str) -> dict:
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