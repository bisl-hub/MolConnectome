import numpy as np
import pandas as pd
import sys
import os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.environ['OPENBLAS_NUM_THREADS'] = '1'

sys.path.append(os.path.join(PROJECT_ROOT, 'src'))


def label_shuffler(control_expression_matrix: pd.DataFrame, schizo_expression_matrix: pd.DataFrame,
                   shuffling_seed: 'str | int'):
    shuffle_seed = int(shuffling_seed) if isinstance(
        shuffling_seed, str) else shuffling_seed
    this_shuffle_rng = np.random.default_rng(seed=shuffle_seed)
    whole_R1_data = pd.concat(
        [control_expression_matrix, schizo_expression_matrix], axis=1)
    shuffled_R1_data_array = this_shuffle_rng.permutation(
        whole_R1_data.to_numpy(), axis=1)
    shuffled_R1_data = pd.DataFrame(data=shuffled_R1_data_array,
                                    index=whole_R1_data.index,
                                    # 익명화
                                    columns=['S' + str(x+1) for x in range(shuffled_R1_data_array.shape[1])])
    common_samples = shuffled_R1_data.columns
    # 대충 반으로 나누기.
    control_samples = common_samples[:(common_samples.shape[0] // 2)]
    schizo_samples = common_samples[(common_samples.shape[0] // 2):]
    shuffled_R1_control = shuffled_R1_data[control_samples]
    shuffled_R1_schizo = shuffled_R1_data[schizo_samples]
    return (shuffled_R1_control, shuffled_R1_schizo)
