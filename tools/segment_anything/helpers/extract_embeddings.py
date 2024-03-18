# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the Apache-2.0 license found in the LICENSE file in the root directory of segment_anything repository and source tree.
# Adapted from onnx_model_example.ipynb in the segment_anything repository.
# Please see the original notebook for more details and other examples and additional usage.
import os
import argparse
import cv2
from tqdm import tqdm
import numpy as np
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..", "..")))

from tools.segment_anything import sam_model_registry, SamPredictor


def main(checkpoint_path, model_type, device, images_folder, embeddings_folder):
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    for image_name in tqdm(os.listdir(images_folder)):
        image_path = os.path.join(images_folder, image_name)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


        predictor.set_image(image)

        image_embedding = predictor.get_image_embedding().cpu().numpy()

        out_path = os.path.join(embeddings_folder, os.path.splitext(image_name)[0] + ".npy")
        # if not os.path.exists(out_path):
        np.save(out_path, image_embedding)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-path", type=str, default="/data/yuqi/code/GP-NeRF-semantic/tools/segment_anything/sam_vit_h_4b8939.pth")
    parser.add_argument("--model_type", type=str, default="default")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--rgbs_path", type=str, default="./example_dataset")
    args = parser.parse_args()

    checkpoint_path = args.checkpoint_path
    model_type = args.model_type
    device = args.device
    images_folder = args.rgbs_path


    embeddings_folder = os.path.join(images_folder, "../sam_features")
    if not os.path.exists(embeddings_folder):
        os.makedirs(embeddings_folder)

    main(checkpoint_path, model_type, device, images_folder, embeddings_folder)
