import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import os
import glob2 
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
        min_index = np.where(remaining_index > 0)[0][np.argmin(distance)]
        sorted_index.append(min_index)
        i = min_index
        remaining_index[i] = 0
    return points[np.array(sorted_index)]



index = 1
colors = sns.color_palette('Paired')
def plot_points(dir, label, style='-*', color='b', shift=[0,0], txt_color='black', normalize_path=None, reverse=True):
    threshold = 0.01
    desired_scores = []
    obtained_scores = []

    paths = [os.path.abspath(path) for path in glob2.glob(os.path.join(dir, '*.csv'))]
    paths += [os.path.abspath(path) for path in glob2.glob(os.path.join(dir, '*', '*.csv'))]

    pref_lis = []
    for path in paths:
        if '.csv' in path:
            full_path = path 
            data = pd.read_csv(full_path)
            # morlhf has less points, let the threshold larger to make the frontier better
            if 'ppo' in path and len(paths) <= 5:
                threshold = 0.5
            obtained_scores.append([np.mean(data['obtained_score1']), np.mean(data['obtained_score2'])])
            if 'pref' in path:
                # get the preference
                if 'eval_data_pref' in path:
                    pref = path.split('eval_data_pref')[-1].strip().split('_')[0]
                    pref_lis.append(float(pref))

    print(pref_lis)
    desired_scores = np.array(desired_scores)
    obtained_scores = np.array(obtained_scores)

    if normalize_path is not None:
        norm_info = np.load(normalize_path)
        norm_info = np.array(norm_info).reshape(2, 2)
        for i in range(2):
            obtained_scores[:, i] = (obtained_scores[:, i] - norm_info[i][0]) / norm_info[i][1] 

    global index
    markersize = 10  if ('*' in style or 'o' in style) else 9
    pareto_points = find_pareto_points(obtained_scores, threshold)
    plt.scatter(obtained_scores[:, 0], obtained_scores[:, 1], marker=style[-1], color=colors[index], s=markersize + 60)
    if len(pref_lis):
        for i in range(len(obtained_scores)):
            plt.annotate('{}'.format(round(pref_lis[i], 1)), (obtained_scores[i, 0] + shift[0], obtained_scores[i, 1] + shift[1]), size=4, color=txt_color)

    plt.plot(pareto_points[:, 0], pareto_points[:, 1], style, c=colors[index], markersize=markersize, label=label)
    index += 2


plt.figure(figsize=(5, 4))

name1 = 'harmless'
name2 = 'helpful'

### replace the paths to your own paths
plot_points('./logs_trl/eval_pretrained', 'Llama 2 base', '*')
plot_points('./logs_trl/eval_sft_alldata', 'SFT', '*')
plot_points('./eval_ppo_pref/', 'MORLHF', '--D', shift=[-0.012, -0.022])
plot_points('./logs_ppo/eval_pposoups_llamma2_klreg0.2', 'Rewarded Soups', style='--s', shift=[-0.012, -0.022])
plot_points('.logs_trl/evalnew_onlinefix_helpful_harmlesshelpful_iter2',  'RiC', style='-o', shift=[-0.012, -0.022], txt_color='white')


plt.xlabel('$R_1$ ({})'.format(name1), fontsize=12)
plt.ylabel('$R_2$ ({})'.format(name2), fontsize=12)
plt.legend(fontsize=11, loc='lower left')
plt.tight_layout()
plt.savefig('ric_assistant_{}_{}.pdf'.format(name1, name2))


