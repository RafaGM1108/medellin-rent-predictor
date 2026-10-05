from medellin_rent.geo.boundaries import LAYERS, download_layer
from medellin_rent.utils.config import get_config

if __name__ == "__main__":
    for name in LAYERS:
        download_layer(name, get_config().paths.raw_geo)
