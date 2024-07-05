import requests
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

"""Script to precache response by multithread hits for the dates requested 
from users for faster response on duplicate test dates"""

url = "http://127.0.0.1:5000/api/predict"  # local host url
start_date = datetime.strptime("02102019", "%d%m%Y")


def process_response(response):
    if response.ok:
        data = response.json()
        # process the data here
    else:
        print(f"Request failed with status code {response.status_code}")


with ThreadPoolExecutor(max_workers=10) as executor:
    futures = []
    for i in range(15):
        date = start_date + timedelta(days=i)
        params = {"date": date.strftime("%d%m%Y")}
        future = executor.submit(requests.get, url, params=params)
        futures.append(future)
    for future in as_completed(futures):
        process_response(future.result())
