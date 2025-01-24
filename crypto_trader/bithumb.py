import jwt
import uuid
import hashlib
import time
from urllib.parse import urlencode
import requests
import yaml
import pandas as pd


# Function to read YAML file
def load_yaml(file_path):
    with open(file_path, "r") as file:
        return yaml.safe_load(file)


# Read the config.yaml file
import os

print(os.getcwd())
config_path = os.path.join("crypto_trader", "config", "credentials.yaml")
credentials = load_yaml(config_path)
api_keys = credentials["api_keys"]["bithumb"]


import requests

url = "https://api.bithumb.com/public/assetsstatus/multichain/ALL"
headers = {"accept": "application/json"}
response = requests.get(url, headers=headers)
response_dict = response.json()
currency_list = response_dict["data"]
pd.DataFrame(currency_list)
print(response.text)


import requests

url = "https://api.bithumb.com/public/orderbook/ALL_KRW"
headers = {"accept": "application/json"}
response = requests.get(url, headers=headers)
response_dict = response.json()

import pprint

pprint.pprint(response_dict["data"])

response.text

print(response.text)
