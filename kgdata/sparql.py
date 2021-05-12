import abc

import requests as rq

from . import util


class Endpoint:
    def __init__(self, url):
        self.url = url


class Wikidata(Endpoint):
    def __init__(self):
        super().__init__("https://query.wikidata.org/sparql")

    def query(self, query):
        response = rq.post(
            self.url,
            params={"format": "json"},
            headers={
                "Content-Type": "application/sparql-query",
                "Accept": "application/json",
            },
            data=query,
        )

        return WikidataResult(response.json())


class WikidataResult:
    def __init__(self, data):
        self.data = data

    @util.cached_property
    def bindings(self):
        return self.data["results"]["bindings"]
