import xml.etree.ElementTree as ET
import requests

def dbpedia_lookup(city_name):
    url = f"https://lookup.dbpedia.org/api/search?query={city_name}&maxResults=50"
    response = requests.get(url, headers={'Accept': 'application/xml'})
    results = []
    if response.status_code == 200:
        root = ET.fromstring(response.content)
        for result in root.findall('./Result'):
            label = result.find('Label').text if result.find('Label') is not None else 'No label'
            description = result.find('Description').text if result.find('Description') is not None else 'No description'
            # TODO: add dbpedia entity type
            results.append([label, description, []])
    return results


if __name__ == '__main__':
    city_to_lookup = "Berlin"
    print(len(dbpedia_lookup(city_to_lookup)))
