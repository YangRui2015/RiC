import os
import seaborn as sns
import glob
import tqdm
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import sys
sys.path.append("/mnt/aigc_cq/private/amandaaluo/own_code/multi_objective/diffusers")
from examples.text_to_image.infer_text_to_image_with_preference_batch import get_samples_from_trainset
from image_metric import ImageMetric
from visualize_pareto import plot_points



f = open("/mnt/aigc_cq/private/amandaaluo/dataset/multi_obj/coco_test_1k.txt", 'r')
data = f.readlines()
VALIDATION_PROMPTS = [d.strip("\n") for d in data]


def get_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--rm1", type=str, help="The reward_model_1"
    )
    parser.add_argument(
        "--rm2", type=str, help="The reward_model_2"
    )
    parser.add_argument(
        "--device", type=int, help="index of GPU"
    )
    parser.add_argument(
        "--step", type=int, help="step"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    device = f"cuda:{args.device}"
    json_file = "/mnt/aigc_cq/private/amandaaluo/dataset/multi_obj/multi_obj_with_clip_base.json"
    rm1, rm2 = args.rm1, args.rm2
    rm1_samples, rm2_samples, mu1, sigma1, mu2, sigma2 = get_samples_from_trainset(json_file, key1=rm1, key2=rm2)
    prompts = VALIDATION_PROMPTS

    
    # calculate preferences and scores
    N = 10
    preferences = np.zeros((N+1, 2))
    preferences[:, 0] = np.arange(0,1 + 1/N, 1/N)
    preferences[:, 1] = 1 - preferences[:, 0]
    preferences = np.round(preferences, 1)
    pareto_points = np.zeros((N+1, 2))
    print(preferences)
    
    scorer = ImageMetric(rm1, rm2, device)

    for k in range(len(preferences)): 
        preference = preferences[k]
        target_rewards = np.zeros(2)
        if preference[0] >= preference[1]:
            target_rewards[0] = np.round(np.quantile(rm1_samples, 0.9999999), 1)
            target_rewards[1] = np.round(np.quantile(rm2_samples, 2 * preference[1]), 1)
        else:
            target_rewards[1] = np.round(np.quantile(rm2_samples, 0.9999999), 1)
            target_rewards[0] = np.round(np.quantile(rm1_samples, 2 * preference[0]), 1)

        print(k, target_rewards)
        

        result_dir = f"results/{rm1}_{rm2}/step-{args.step}/imgs/{rm1}_{target_rewards[0]}_{rm2}_{target_rewards[1]}"
        

        filenames = glob.glob(os.path.join(result_dir, "*.png"))
        filenames = sorted(filenames, key=lambda x: int(os.path.basename(x).split(".")[0].split("-")[1]))

        # print(filenames)
        assert len(filenames) == len(prompts)

        idx = 0
        r1_scores = []
        r2_scores = []
        for i in tqdm.tqdm(range(len(filenames))):            
            img_name = filenames[i]
            image = Image.open(img_name).convert("RGB")
            r1_score, r2_score = scorer.get_score(prompts[i], image)
            r1_scores.append(r1_score)
            r2_scores.append(r2_score)
        
        avg_r1 = sum(r1_scores) / len(r1_scores)
        avg_r2 = sum(r2_scores) / len(r2_scores)

        pareto_points[k, 0] = avg_r1
        pareto_points[k, 1] = avg_r2
    
    # scale
    pareto_points[:, 0] = (pareto_points[:, 0] - mu1) / sigma1 
    pareto_points[:, 1] = (pareto_points[:, 1] - mu2) / sigma2     

    output_dir = f"results/{rm1}_{rm2}/step-{args.step}/metrics"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    np.savetxt(os.path.join(output_dir, 'pareto_points.txt'), pareto_points)

    x,y = pareto_points[:, 0], pareto_points[:, 1]
    plt.scatter(x, y, marker='o')
    plt.savefig(os.path.join(output_dir, "show_point.png"))
    plt.close()
    plt.plot(x, y, marker='o')
    plt.savefig(os.path.join(output_dir, "show_trend.png"))
    plt.close()

    plot_points(pareto_points, label="aes_jpg_coco", style='-o', shift=[-0.01,-0.002], txt_color='white', output_dir=output_dir)


    