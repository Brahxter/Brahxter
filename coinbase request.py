from coinbase.wallet.client import Client
import requests
import hashlib  # For password hashing


api_key = "3c0f8e2c-50c2-462a-b0fb-ff02e8b45331"
api_secret = "\nMHcCAQEEIMyQZh2rMhYSskz9CtGcP0ftDW2DjOyeGpga6CKZbTGloAoGCCqGSM49\nAwEHoUQDQgAEqev8+ouR6VwGQTvL4sPq5gRzNF9P871ZqV8MfohTGx7hS0YB6Fy1\nEW2ucn5rr0WZ/qY0nwrk/6uxCtkBdESSpw==\n"

client = Client(api_key, api_secret)

user = client.get_current_user()