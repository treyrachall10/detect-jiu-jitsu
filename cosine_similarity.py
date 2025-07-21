import pandas as pd
import numpy as np
import consts
import torch
import ast
from utils import print_cos_ids

def cos_similarity_loop():
    """
    Applies cosine similarity to every vector and assigns id's.

    Returns:
        None: Prints images and their corresponding id's
    """
    df = pd.read_csv(f"csv_output/has_features/features_transformer_bjjvideo.csv")
    features = df['features'].apply(ast.literal_eval).tolist()
    filenames = df['filename'].tolist()

    # Holds feature data we've already looked at
    known_data = []

    current_id = 0

    # Iterate through every feature and its filename in df
    for i, (filename, feature) in enumerate(zip(filenames, features)):

        # Converts feature vector to tensor
        current_feature = torch.tensor(feature, dtype=torch.float32)

        # List of cos similarity values
        similarity_list = []

        if i == 0:
            known_data.append({
                'filename': filename,
                'feature': current_feature,
                'id': str(current_id)
            })
        else:

            # Compare current feature to all known features using cosine similarity
            for seen_feature in known_data:
                cos_sim = torch.nn.CosineSimilarity(dim=0)
                similarity = cos_sim(current_feature, seen_feature['feature'])
                similarity_list.append(similarity)
        
            # Assign same id from most similar value if most similar value is above certain threshold(.825) otherwise give own id 
            most_sim = max(similarity_list)

            if most_sim > consts.threshold_value:
                indx_most_similar = similarity_list.index(most_sim)
                id_num = known_data[indx_most_similar]['id']
                known_data.append({
                    'filename': filename,
                    'feature': current_feature,
                    'id': id_num
                })
            else:
                known_data.append({
                    'filename': filename,
                    'feature': current_feature,
                    'id': str(current_id)
                })
                current_id += 1

    print_cos_ids(known_data=known_data)

cos_similarity_loop()

    
