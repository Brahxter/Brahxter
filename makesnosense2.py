import random
import hashlib
from coinbase.wallet.client import Client
import requests



class WhimsicalToRealWallet:
    def __init__(self, password):
        self.password_hash = hashlib.sha256(password.encode()).hexdigest()
        self.imaginary_balance = 0.0
        self.real_balance = 0.0
        self.transactions = []

    def add_random_fraction(self):
        # Generate a random fraction between 0 and 1 (exclusive)
        fraction = random.random()
        # Convert to cents (multiply by 100)
        cents = fraction * 100
        # Add to the imaginary balance
        self.imaginary_balance += cents
        self.transactions.append(f"Added {cents:.2f} imaginary cents")

    def convert_to_real_usd(self):
        # Convert imaginary cents to real USD (divide by 100)
        real_usd = self.imaginary_balance / 100
        # Add to the real balance
        self.real_balance += real_usd
        self.transactions.append(f"Converted {real_usd:.2f} USD")

    def store_in_coinbase_wallet(self):
        # Assume you have a function to send funds to Coinbase Wallet
        # This is purely imaginary!
        # In reality, you'd need to interact with Coinbase's API.
        # For now, let's pretend it works like this:
        # (Replace this with actual Coinbase Wallet integration)
        import requests  # Add missing import statement

        
        # Your Coinbase API credentials (replace with your actual values)
        api_key = "JJbzfgpOQe6I9myG"
        api_secret = "vYTWlUpjvJLWnVimEKsbtyHjxoOMEIzc"

            # Construct the API endpoint
        url = "https://api.coinbase.com/v2/accounts"  # Example endpoint

        

            # Create a request with authentication headers
        headers = {
                "Authorization": f"Bearer {api_key}",
                "CB-VERSION": "2022-08-24",  # API version
            }

            # Make a GET request to retrieve account info
        response = requests.get(url, headers=headers)
        data = response.json()
            
            # Extract relevant info (e.g., account ID, balance, etc.)
        account_id = data["data"][0]["id"]
        account_balance = data["data"][0]["balance"]["amount"]

            # Simulate sending funds (replace with actual logic)
        print(f"Sending {self.real_balance:.2f} USD to your Coinbase Wallet (Account ID: {account_id})... Done!")

        

        print(f"Sending {self.real_balance:.2f} USD to your Coinbase Wallet... Done!")

    def display_balances(self):
        print(f"Imaginary balance: ${self.imaginary_balance:.2f}")
        print(f"Real balance: ${self.real_balance:.2f}")

    def verify_password(self, input_password):
        input_hash = hashlib.sha256(input_password.encode()).hexdigest()
        return input_hash == self.password_hash

# Create our whimsical-to-real wallet
password = input("Enter your whimsical password: ")
my_whimsical_wallet = WhimsicalToRealWallet(password)

# Simulate adding random fractions daily (you can adjust the frequency)
for _ in range(30):  # Let's simulate 30 days
    my_whimsical_wallet.add_random_fraction()

# Convert imaginary cents to real USD
my_whimsical_wallet.convert_to_real_usd()

# Store in Coinbase Wallet (imaginary function)
my_whimsical_wallet.store_in_coinbase_wallet()

# Display the final balances
my_whimsical_wallet.display_balances()

# Verify whimsical password (just for fun)
input_password = input("Enter your whimsical password to verify: ")
if my_whimsical_wallet.verify_password(input_password):
    print("Whimsical password verified! Access granted.")
else:
    print("Incorrect whimsical password. Access denied.")