from medellin_rent.data.listings import RAW_FILE, SOURCE_URL, load_listings, record_source
from medellin_rent.utils.config import get_config

if __name__ == "__main__":
    path = get_config().paths.raw_listings / RAW_FILE
    if not path.is_file():
        raise SystemExit(f"Download {RAW_FILE} from {SOURCE_URL} into {path.parent}")
    record_source(path)
    load_listings(path)
