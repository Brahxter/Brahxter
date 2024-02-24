from coinbase.wallet.client import Client
import requests
import hashlib  # For password hashing


api_key = "JJbzfgpOQe6I9myG"
api_secret = "vYTWlUpjvJLWnVimEKsbtyHjxoOMEIzc"

client = Client(api_key, api_secret)

user = client.get_current_user()