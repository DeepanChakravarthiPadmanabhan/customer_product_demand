import requests
from datetime import datetime, timedelta

"""Script to precache response for the dates requested from users for faster
 response on duplicate test dates"""

url = "http://127.0.0.1:5000/api/predict"  # local host url
date = datetime.strptime("18102019", "%d%m%Y")

for i in range(5):
    params = {"date": date.strftime("%d%m%Y")}
    response = requests.get(url, params=params)
    if response.ok:
        data = response.json()
        # process the data here
    else:
        print(f"Request failed with status code {response.status_code}")
    date += timedelta(days=1)
