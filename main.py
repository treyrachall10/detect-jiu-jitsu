import argparse
import consts
import os
from pathlib import Path

from model.model_factory import get_model
from model.supported_extractors import model_supported

from detect_crop import make_detections
from utils import normalize_path
from cluster.algo_factory import get_cluster_algo
from cluster.cluster import cluster
from feature_extraction import extract_features

def create_csv_dirs():
    os.makedirs(f"crops/{consts.video_name}", exist_ok=True)
    os.makedirs("csv_output/no_features", exist_ok=True)
    os.makedirs("csv_output/has_features", exist_ok=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tool for processing each id in video")
    parser.add_argument('--extractor', type=str, help="Name of feature extractor", required=True)
    parser.add_argument('--video_path', type=str, help="Path to video file being processed", required=True)
    parser.add_argument('--cluster_algo', type=str, help="Clustering algorithm for data", required=True)
    args = parser.parse_args()

    consts.video_path = normalize_path(args.video_path)

    consts.video_name = Path(consts.video_path).stem
    
    cluster_algo = get_cluster_algo(args.cluster_algo)

    # Create output directory for crops
    create_csv_dirs()

    # Assigns model and global variables if model is supported
    if model_supported(args.extractor):
        consts.extractor = get_model(args.extractor)
        consts.extractor_name = args.extractor
        consts.metadata_file_name = f"metadata_{consts.extractor_name}_{consts.video_name}.csv"
        consts.features_file_name = f"features_{consts.extractor_name}_{consts.video_name}.csv"

    # Make detections and save csv if csv file for video doesn't exist
    #if all(config.metadata_file_name not in file.name for file in config.NO_FEATURES_PATH.iterdir()):
        # make_detections()
    if all(consts.features_file_name not in file.name for file in consts.FEATURES_PATH.iterdir()):
        extract_features()
    
    cluster(algo=cluster_algo)