import urllib.request
import os
import logging
import ssl

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def download_data(data_dir="data"):
    """
    Downloads NASA GISS and NOAA ONI data to the specified directory.
    """
    os.makedirs(data_dir, exist_ok=True)

    # NASA GISS URL might be flaky or restricted.
    # Try alternate or mirror if available.
    # https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv
    urls = {
        "GLB.Ts+dSST.csv": "https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv",
        "oni.ascii.txt": "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt",
    }

    headers = {'User-Agent': 'Mozilla/5.0'}

    # SSL Context for unverified if needed (though NASA usually has valid certs)
    context = ssl._create_unverified_context()

    for filename, url in urls.items():
        logging.info(f"Downloading {filename} from {url}...")
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, context=context, timeout=30) as response:
                data = response.read()
                filepath = os.path.join(data_dir, filename)
                with open(filepath, 'wb') as f:
                    f.write(data)
            logging.info(f"Successfully downloaded {filename} to {filepath}")
        except Exception as e:
            logging.error(f"Failed to download {filename}: {e}")
            # If GISTEMP fails, try to see if we have a local copy from previous steps or handle gracefully?
            # In a real environment, we would retry or fail.
            # Here, if it fails, the pipeline will break.
            pass

if __name__ == "__main__":
    download_data()
