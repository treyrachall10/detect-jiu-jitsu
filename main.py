import argparse
import config
import os
from pathlib import Path
from model_factory import get_model
from supported_models import model_supported
from detectncrop import make_detections, detect_crop_write
from utils import normalize_path
from features import extract_features
from cluster_algo_factory import get_cluster_algo
from cluster import cluster

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tool for processing each id in video")
    parser.add_argument('--extractor', type=str, help="Name of feature extractor", required=True)
    parser.add_argument('--video_path', type=str, help="Path to video file being processed", required=True)
    parser.add_argument('--cluster_algo', type=str, help="Clustering algorithm for data", required=True)
    args = parser.parse_args()

    config.video_path = normalize_path(args.video_path)
    config.video_name = Path(config.video_path).stem
    cluster_algo = get_cluster_algo(args.cluster_algo)

    # Create output directory for crops
    os.makedirs(f"crops/{config.video_name}", exist_ok=True)
    os.makedirs("csv_output/no_features", exist_ok=True)
    os.makedirs("csv_output/has_features", exist_ok=True)

    # Assigns model if model is supported
    if model_supported(args.extractor):
        config.extractor = get_model(args.extractor)
        config.extractor_name = args.extractor
        config.metadata_file_name = f"metadata_{config.extractor_name}_{config.video_name}.csv"
        config.features_file_name = f"features_{config.extractor_name}_{config.video_name}.csv"
    else:
        raise ValueError(f"Model '{args.extractor}' is not supported")

    # Make detections and save csv if csv file for video doesn't exist
    if all(config.metadata_file_name not in file.name for file in config.NO_FEATURES_PATH.iterdir()):
        results = make_detections()
        detect_crop_write(results)
    if all(config.features_file_name not in file.name for file in config.FEATURES_PATH.iterdir()):
        extract_features()
    
    cluster(algo=cluster_algo)