import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import os
import seaborn as sns

def find_pareto_points(obtained_scores, threshold=0.02):
    n = len(obtained_scores)
    if n == 1:
        return obtained_scores
    pareto_index = []
    high_low = np.max(obtained_scores, axis=0) - np.min(obtained_scores, axis=0)
    for i in range(n):
        if not any(np.all((obtained_scores - obtained_scores[i] - threshold * high_low) > 0.0, axis=1)):
            pareto_index.append(i)

    points = obtained_scores[np.array(pareto_index)]
    arg_index = np.argsort(points[:, 0])
    points = points[arg_index]
    print(points)
    sorted_index = [0]
    remaining_index = np.ones(len(points))
    i = 0
    remaining_index[i] = 0
    while sum(remaining_index):
        distance = ((points[np.where(remaining_index)] - points[i]) ** 2 ).sum(axis=1)
        # print(distance)
        min_index = np.where(remaining_index > 0)[0][np.argmin(distance)]
        sorted_index.append(min_index)
        i = min_index
        remaining_index[i] = 0
        # print(remaining_index)
        # import pdb;pdb.set_trace()
    return points[np.array(sorted_index)]



colors = sns.color_palette('Paired')
index = 9

# for jpeg-aes
def plot_points(obtained_scores, label, style='-*', color='b', shift=[0,0], txt_color='black', normalize_path=None, reverse=True, output_dir=None):
    threshold = 0.01
    pref_lis = list(np.arange(0, 1.1, 0.1))
    print(pref_lis)

    global index
    index = 1

    # baseline: aes, jepg
    plt.scatter(-1.838043551710084600e-01, -1.038899692290953247e+00, marker='*', color=colors[index], s=70, label='SD1.5 base')

    markersize = 10  if ('*' in style or 'o' in style) else 9
    index = 9
    pareto_points = find_pareto_points(obtained_scores, threshold)
    plt.scatter(obtained_scores[:, 0], obtained_scores[:, 1], marker=style[-1], color=colors[index], s=markersize + 60)
    if len(pref_lis):
        for i in range(len(obtained_scores)):
            plt.annotate('{}'.format(round(pref_lis[i], 1)), (obtained_scores[i, 0] + shift[0], obtained_scores[i, 1] + shift[1]), size=5, color=txt_color)

    plt.plot(pareto_points[:, 0], pareto_points[:, 1], style, c=colors[index], markersize=markersize, label=label)
    index += 2

    plt.legend(loc='lower left')

    plt.xlabel('$R_1$ (aesthetic)')
    plt.ylabel('$R_1$ (compressible)')


    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_name = os.path.join(output_dir, 'pareto_with_baseline.png')
    plt.savefig(output_name)


