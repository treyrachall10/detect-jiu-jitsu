"""
    Universal global variables
"""
from utils import normalize_path

NO_FEATURES_PATH = normalize_path("csv_output/no_features")
FEATURES_PATH = normalize_path("csv_output/has_features")
video_name = None
video_path = None
extractor = None
extractor_name = None
metadata_file_name = None
features_file_name = None
processor = None

threshold_value = .825