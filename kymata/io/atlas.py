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
    """
    response = requests.get(api)
    return json.loads(response.text)
