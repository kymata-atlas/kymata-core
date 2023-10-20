import json
from enum import Enum, auto
from typing import NamedTuple

import requests


class InputStream(Enum):
    visual = auto()
    auditory = auto()
    tactile = auto()


class KymataDataset:
    pass


class FunctionMetadata(NamedTuple):
    kid: str
    name: str
    overview: str
    # equation:
    input_stream: InputStream
    # ref:
    # datasets: list[KymataDataset]
    tags: list[str]

    @property
    def href(self) -> str:
        return f"/api/functions/{self.kid}"


class API:
    endpoint_url = "https://kymata.org/api/"

    def get_function_metadata(self, kid: str) -> FunctionMetadata:
        query_url = API.endpoint_url + "functions/" + kid
        response = requests.get(query_url)
        response_dict = json.loads(response.text)

        return FunctionMetadata(
            kid=response_dict["kid"],
            name=response_dict["name"],
            overview=response_dict["overview"],
            input_stream=InputStream[response_dict["input_stream"].lower()],
            tags=response_dict["tags"],
        )

    def 


if __name__ == "__main__":
    function_meta = API().get_function_metadata("FPWMD")

    print(function_meta.name)
