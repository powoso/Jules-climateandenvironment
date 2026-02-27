import urllib.request
import os

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

urls = {
    "GLB.Ts+dSST.csv": "https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv",
    "oni.ascii.txt": "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt",
    # "N_seaice_extent_daily_v3.0.csv": "https://masie_web.apps.nsidc.org/pub/DATASETS/NOAA/G02135/north/daily/data/N_seaice_extent_daily_v3.0.csv"
    # The NSIDC one seems flaky or requires specific headers. I'll skip it for now or try a different approach.
}

# Alternative source for sea ice if possible, or I will use a placeholder.
# For now let's try to get the first two.

headers = {'User-Agent': 'Mozilla/5.0'}

for filename, url in urls.items():
    print(f"Downloading {filename} from {url}...")
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req) as response:
            data = response.read()
            with open(os.path.join(DATA_DIR, filename), 'wb') as f:
                f.write(data)
        print(f"Successfully downloaded {filename}")
    except Exception as e:
        print(f"Failed to download {filename}: {e}")
