import urllib.request
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def download_data(data_dir="data"):
    """
    Downloads NASA GISS and NOAA ONI data to the specified directory.
    """
    os.makedirs(data_dir, exist_ok=True)

    urls = {
        "GLB.Ts+dSST.csv": "https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv",
        "oni.ascii.txt": "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt",
    }

    headers = {'User-Agent': 'Mozilla/5.0'}

    for filename, url in urls.items():
        logging.info(f"Downloading {filename} from {url}...")
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req) as response:
                data = response.read()
                filepath = os.path.join(data_dir, filename)
                with open(filepath, 'wb') as f:
                    f.write(data)
            logging.info(f"Successfully downloaded {filename} to {filepath}")
        except Exception as e:
            logging.error(f"Failed to download {filename}: {e}")

if __name__ == "__main__":
    download_data()
