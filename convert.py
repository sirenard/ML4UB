import gzip
import os
import pickle
from pathlib import Path

import numpy as np

from model import get_sample_files


def convert(scores, alpha):
    indexes = np.where(np.isfinite(scores))
    m = np.max(scores[indexes])

    scores[indexes] = np.int64(scores[indexes] >= (1-alpha)*m)
    return scores

def convert2(scores, alpha):
    indexes = np.where(np.isfinite(scores))

    s = 1 + scores[indexes] / np.linalg.norm(scores[indexes])
    m = np.max(s)

    if alpha is not None:
        scores[indexes] = np.int64(s >= (1 - alpha) * m)
    return scores

def convert_scores(folder, output_folder, conversion_function, *args, **kwargs):
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    files = get_sample_files(folder)

    for file in files:
        with gzip.open(file, "rb") as f:
            observation, action, action_set, scores = pickle.load(f)

        scores = conversion_function(scores, *args, **kwargs)

        _, filename = os.path.split(file)
        with gzip.open(os.path.join(output_folder, filename), "wb") as f:
            data = [observation, action, action_set, scores]
            pickle.dump(data, f)


if __name__ == "__main__":
    convert_scores("ub_0_1_limit_20", "ub_0_1_limit_20_alpha2_0.1", convert2, alpha=0.1)


