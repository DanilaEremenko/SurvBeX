from typing import List, Optional

import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE


def draw_surv_yo(time_points, pred_surv, actual_et: List[float], draw_args: Optional[List[dict]] = None):
    for i, surv_func in enumerate(pred_surv):
        curr_draw_args = draw_args[i] if draw_args is not None else {}
        p = plt.step(time_points, surv_func(time_points), where="post", **curr_draw_args)
        plt.plot([actual_et[i], actual_et[i]], [0, 1], '--', color=p[0].get_color(), )

    plt.ylabel("SF")
    plt.xlabel("T")


def draw_points_tsne(pt_groups: List[np.ndarray], names: List[str], colors: List[str], path: str):
    assert len(pt_groups) == len(names) == len(colors)
    all_features = np.vstack(pt_groups)
    all_tsne = TSNE(perplexity=min(100, len(all_features) - 1)).fit_transform(all_features)
    pt_groups_tsne = []
    lb = 0
    for pt_group in pt_groups:
        pt_groups_tsne.append(all_tsne[lb:lb + len(pt_group)])
        lb += len(pt_group)

    for pt_group_tsne, name, color in zip(pt_groups_tsne, names, colors):
        plt.scatter(pt_group_tsne[:, 0], pt_group_tsne[:, 1], label=name, color=color)

    plt.legend()
    plt.savefig(path)
    plt.clf()
