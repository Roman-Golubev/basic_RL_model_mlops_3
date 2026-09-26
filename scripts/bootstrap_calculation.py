import os
import numpy as np
import pandas as pd

from scripts.rl_model import RL_func
from scripts.pipeline_config import DATA_INPUT_DIR, DATA_OUTPUT_DIR, API_CONFIGS, API_NUM


def bootstrap_calculation(api_num=API_NUM, epsilon=0.7, min_epsilon=0.2, decay_rate=0.992):
    cfg = API_CONFIGS[api_num]
    input_path = os.path.join(DATA_INPUT_DIR, cfg["params_file"])
    output_dir = os.path.join(DATA_OUTPUT_DIR, cfg["output_dir"])
    output_path = os.path.join(output_dir, cfg["output_file"])

    params_df = pd.read_csv(input_path)
    csv_path = os.path.join(DATA_INPUT_DIR, "fins.csv")
    fins_df = pd.read_csv(csv_path)

    # 50 итераций, распределяемых по эпизодам
    for _ in range(50):
        results_dict = {
            'Q_s_a': None, 'tem_distr': None, 'fin_num': None, 'Pus_prev': None,
            'total_reward': None, 'steps_total': None, 's_next': None, 'done': None,
            'episode': None,
            'states_lst': None, 'actions_lst': None, 'rewards_lst': None, 'del_Pus_lst': None,
            'Pus_final': None, 'tem_distr_final': None, 'fin_num_final': None,
            'step_num_final': None
        }

        output_df = pd.read_pickle(output_path)
        episode = output_df.loc[len(output_df)-1, 'episode']
        steps_total = output_df.loc[len(output_df)-1, 'steps_total']
        done = output_df.loc[len(output_df)-1, 'done']
        # продолжение текущего эпизода
        if steps_total < 450 and not done:
            # начало первого эпизода
            if steps_total == 0:
                total_reward = 0
                Pus_prev = 0
                s_next = 0
                Pus_final = -1
                tem_distr = np.array([100] * 10 * 3).reshape(3, 10)
                fin_type = fins_df.iloc[0]['fin_type']
                fin_num = fins_df[fins_df['fin_type'] == fin_type].index[0]
            # продолжение эпизода
            else:
                total_reward = output_df.loc[len(output_df)-1, 'total_reward']
                Pus_prev = output_df.loc[len(output_df)-1, 'Pus_prev']
                s_next = output_df.loc[len(output_df)-1, 's_next']
                Pus_final = output_df.loc[len(output_df)-1, 'Pus_final']
                tem_distr = output_df.loc[len(output_df)-1, 'tem_distr'].copy()
                fin_num = output_df.loc[len(output_df)-1, 'fin_num']
        # начало последующих эпизодов
        else:
            episode += 1
            steps_total = 0
            total_reward = 0
            Pus_prev = 0
            s_next = 0
            Pus_final = -1
            tem_distr = np.array([100] * 10 * 3).reshape(3, 10)
            fin_type = fins_df.iloc[0]['fin_type']
            fin_num = fins_df[fins_df['fin_type'] == fin_type].index[0]
        results_dict['episode'] = episode

        # Выполнение шагов RL-агентом (25 шагов или меньше)
        # используется предобученная Q_s_a
        Q = output_df.loc[len(output_df)-1, 'Q_s_a'].copy()
        epsilon_now = max(
            min_epsilon,
            epsilon * decay_rate**(episode - 1)
        )
        (
            results_dict['Q_s_a'], results_dict['tem_distr'], results_dict['fin_num'], results_dict['Pus_prev'],
            results_dict['total_reward'], results_dict['steps_total'], results_dict['s_next'], results_dict['done'],
            results_dict['states_lst'], results_dict['actions_lst'], results_dict['rewards_lst'], results_dict['del_Pus_lst'],
            results_dict['Pus_final'], results_dict['tem_distr_final'], results_dict['fin_num_final'],
            results_dict['step_num_final']
        ) = RL_func(
            params_df.iloc[0], fins_df,
            alpha = 0.2,
            gamma = 1,
            epsilon_now = epsilon_now,
            seed = 42,
            total_reward = total_reward,
            steps_total = steps_total,
            Pus_prev = Pus_prev,
            Q = Q,
            tem_distr = tem_distr,
            fin_num = fin_num,
            s = s_next,
            Pus_final = Pus_final
        )

        # сохранение результатов теущей итерации
        cur_result = pd.DataFrame([results_dict])
        output_df = pd.concat([output_df, cur_result], ignore_index=True)
        output_df.to_pickle(output_path)


if __name__ == "__main__":
    bootstrap_calculation()
