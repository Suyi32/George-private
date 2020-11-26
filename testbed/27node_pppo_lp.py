# other name: PPPOWithoutSubScheduler_PCPO_PPPO_me_mini_sim.py
import numpy as np
import time
import os
import sys

sys.path.append("/Users/ourokutaira/Desktop/George")
sys.path.append("../")
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from testbed.cluster_env import LraClusterEnv
from testbed.PolicyGradient_PCPO_PPO import PolicyGradient
from z3 import *
import time
print("Brain: PolicyGradient_PCPO_PPO")
print("using PPPO")
import argparse
from testbed.simulator.simulator import Simulator
import matplotlib.pyplot as plt

"""
fish
'--batch_choice': 0, 1, 2, ``` 30
python3 PCPOWithoutSubScheduler.py --batch_choice 0
"""

hyper_parameter = {
    'batch_C_numbers': None
}
params = {
    'batch_size': 20,
    # 'epochs': 100000,
    'epochs': 80000,
    'path': "pppo_27_" + str(hyper_parameter['batch_C_numbers']),
    'rec_path': "pppo_separate_unified_replay_level_formal_new100",
    'recover': False,
    'learning rate': 0.01,
    'nodes per group': 3,
    'number of nodes in the cluster': 27,
    'replay size': 100,
    'container_limitation per node': 8
}

NUM_CONTAINERS = 100

app_node_set = np.array([
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26],
    [2, 3, 5, 6, 7, 11, 12, 18, 20, 22, 23, 24, 25, 26],
    [0, 2, 8, 9, 19, 23, 24, 25, 26],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]])
for idx in range(len(app_node_set)):
    print("[INFO] No. of nodes for App {}: {}".format(idx, len(app_node_set[idx])))


def train(params):
    """
    parameters set
    """
    print("Current params", params)

    NUM_NODES = params['number of nodes in the cluster']
    env = LraClusterEnv(num_nodes=NUM_NODES)
    batch_size = params['batch_size']
    ckpt_path_1 = "./checkpoint/" + params['path'] + "1/model.ckpt"
    ckpt_path_2 = "./checkpoint/" + params['path'] + "2/model.ckpt"
    ckpt_path_3 = "./checkpoint/" + params['path'] + "3/model.ckpt"

    ckpt_path_rec_1 = "./checkpoint/" + params['path'] + "1/model.ckpt"
    ckpt_path_rec_2 = "./checkpoint/" + params['path'] + "2/model.ckpt"
    ckpt_path_rec_3 = "./checkpoint/" + params['path'] + "3/model.ckpt"

    np_path = "./checkpoint/" + params['path'] + "/optimal_file_name.npz"
    Recover = params['recover']
    nodes_per_group = int(params['nodes per group'])
    replay_size = params['replay size']
    training_times_per_episode = 1  # TODO: if layers changes, training_times_per_episode should be modified
    # safety_requirement = 2.0 / 100.
    safety_requirement = params['safety_requirement']
    print("######## safety_requirement = {} ########".format(safety_requirement))
    """
    Build Network
    """
    n_actions = nodes_per_group  #: 3 nodes per group
    n_features = int(n_actions * (env.NUM_APPS + 1 + env.NUM_APPS) + 1 + env.NUM_APPS)  #: 3*9+1 = 28
    RL_1 = PolicyGradient(
        n_actions=n_actions,
        n_features=n_features,
        learning_rate=params['learning rate'],
        suffix=str(100) + '1a',
        safety_requirement=safety_requirement,
        params=params)

    RL_2 = PolicyGradient(
        n_actions=n_actions,
        n_features=n_features,
        learning_rate=params['learning rate'],
        suffix=str(100) + '2a',
        safety_requirement=safety_requirement,
        params=params)

    RL_3 = PolicyGradient(
        n_actions=n_actions,
        n_features=n_features,
        learning_rate=params['learning rate'],
        suffix=str(100) + '3a',
        safety_requirement=safety_requirement,
        params=params)

    sim = Simulator()

    """
    Training
    """
    start_time = time.time()
    global_start_time = start_time
    number_optimal = []
    observation_episode_1, action_episode_1, reward_episode_1, safety_episode_1 = [], [], [], []
    observation_optimal_1, action_optimal_1, reward_optimal_1, safety_optimal_1 = [], [], [], []

    observation_episode_2, action_episode_2, reward_episode_2, safety_episode_2 = [], [], [], []
    observation_optimal_2, action_optimal_2, reward_optimal_2, safety_optimal_2 = [], [], [], []

    observation_episode_3, action_episode_3, reward_episode_3, safety_episode_3 = [], [], [], []
    observation_optimal_3, action_optimal_3, reward_optimal_3, safety_optimal_3 = [], [], [], []

    epoch_i = 0

    thre_entropy = 0.1
    # TODO: delete this range

    names = locals()
    for i in range(7):
        names['x' + str(i)] = z3.IntVector('x' + str(i), 3)
    for i in range(0, 10):
        names['highest_tput_' + str(i)] = 0
        names['observation_optimal_1_' + str(i)] = []
        names['action_optimal_1_' + str(i)] = []
        names['observation_optimal_2_' + str(i)] = []
        names['action_optimal_2_' + str(i)] = []
        names['observation_optimal_3_' + str(i)] = []
        names['action_optimal_3_' + str(i)] = []
        names['reward_optimal_1_' + str(i)] = []
        names['reward_optimal_2_' + str(i)] = []
        names['reward_optimal_3_' + str(i)] = []
        names['safety_optimal_1_' + str(i)] = []
        names['safety_optimal_2_' + str(i)] = []
        names['safety_optimal_3_' + str(i)] = []
        names['number_optimal_' + str(i)] = []
        names['optimal_range_' + str(i)] = 1.05
        names['lowest_vio_' + str(i)] = 500
        names['observation_optimal_1_vio_' + str(i)] = []
        names['action_optimal_1_vio_' + str(i)] = []
        names['observation_optimal_2_vio_' + str(i)] = []
        names['action_optimal_2_vio_' + str(i)] = []
        names['observation_optimal_3_vio_' + str(i)] = []
        names['action_optimal_3_vio_' + str(i)] = []
        names['reward_optimal_vio_1_' + str(i)] = []
        names['reward_optimal_vio_2_' + str(i)] = []
        names['reward_optimal_vio_3_' + str(i)] = []
        names['safety_optimal_vio_1_' + str(i)] = []
        names['safety_optimal_vio_2_' + str(i)] = []
        names['safety_optimal_vio_3_' + str(i)] = []
        names['number_optimal_vio_' + str(i)] = []
        names['optimal_range_vio_' + str(i)] = 1.1

    def store_episode_1(observations, actions):
        observation_episode_1.append(observations)
        action_episode_1.append(actions)

    def store_episode_2(observations, actions):
        observation_episode_2.append(observations)
        action_episode_2.append(actions)

    def store_episode_3(observations, actions):
        observation_episode_3.append(observations)
        action_episode_3.append(actions)

    tput_origimal_class = 0
    source_batch_, index_data_ = batch_data(NUM_CONTAINERS, env.NUM_APPS)  # index_data = [0,1,2,0,1,2]

    time_ep_acc = 0.0
    time_al_acc = 0.0
    while epoch_i < params['epochs']:
        time_ep_start = time.time()

        if Recover:
            print("Recover from {}".format(ckpt_path_rec_1))
            RL_1.restore_session(ckpt_path_rec_1)
            RL_2.restore_session(ckpt_path_rec_2)
            RL_3.restore_session(ckpt_path_rec_3)
            Recover = False

        observation = env.reset().copy()  # (9,9)
        source_batch = source_batch_.copy()
        index_data = index_data_.copy()



        def handle_constraint(observation_now, appid_now):

            observation_original = observation_now.copy()

            mapping_index = []
            list_check = []

            t2 = time.time()
            for place in range(3):
                s.push()
                s.add(names['x' + str(appid_now)][place] >= int(observation_now[place][appid_now]) + 1)

                if s.check() == z3.sat:
                    list_check.append(False)
                else:
                    list_check.append(True)
                s.pop()

            t3 = time.time()
            # print("formulate: ", t2 - t1)
            # print("calculate: ", t3 - t2)
            good_index = np.where(np.array(list_check) == False)[0]
            length = len(good_index)
            if length < 1:
                test = 1
            index_replace = 0
            for node in range(3):
                if list_check[node]:  # bad node
                    # index_this_replace = good_index[np.random.randint(length)]
                    index_this_replace = good_index[index_replace % length]
                    index_replace += 1
                    observation_original[node] = observation[index_this_replace]
                    mapping_index.append(index_this_replace)
                else:
                    mapping_index.append(node)
                    observation_original[node] = observation[node]

            return observation_original, mapping_index


        """
        Episode
        """
        """
        first layer
        """
        time_al_start = time.time()

        total = source_batch
        limit = (1 * 9)
        capicity = (8 * 9)  # 3
        s = Solver()
        # app sum == batch
        for i in range(7):
            s.add(z3.Sum(names['x' + str(i)]) == int(total[i]))
        # node capacity
        for node in range(3):
            s.add(z3.Sum([names['x' + str(i)][node] for i in range(7)]) <= int(capicity))
        # >=0
        for i in range(7):
            for node in range(3):
                s.add(names['x' + str(i)][node] >= 0)
        # per app spread
        for i in range(7):
            for node in range(3):
                s.add(names['x' + str(i)][node] <= limit)
        # App1 and App2 not exist
        for node in range(3):
            s.add(names['x' + str(1)][node] + names['x' + str(2)][node] <= limit)

        source_batch_first = source_batch_.copy()
        observation_first_layer = np.zeros([nodes_per_group, env.NUM_APPS], int)
        for inter_episode_index in range(NUM_CONTAINERS):
            appid = index_data[inter_episode_index]
            observation_first_layer_copy, mapping_index = handle_constraint(observation_first_layer, appid)
            assert len(mapping_index) > 0

            source_batch_first[appid] -= 1
            # observation_first_layer_copy = observation_first_layer.copy()
            observation_first_layer_copy[:, appid] += 1
            observation_first_layer_copy = np.append(observation_first_layer_copy, observation_first_layer_copy > 9 * 2,
                                                     axis=1)
            observation_first_layer_copy = np.append(observation_first_layer_copy,
                                                     observation_first_layer_copy.sum(axis=1).reshape(nodes_per_group,
                                                                                                      1), axis=1)
            observation_first_layer_copy = np.array(observation_first_layer_copy).reshape(1, -1)
            observation_first_layer_copy = np.append(observation_first_layer_copy, appid).reshape(1, -1)
            observation_first_layer_copy = np.append(observation_first_layer_copy,
                                                     np.array(source_batch_first)).reshape(1, -1)
            action_1, prob_weights = RL_1.choose_action(observation_first_layer_copy.copy())
            decision = mapping_index[action_1]
            observation_first_layer[decision, appid] += 1
            s.add(names['x' + str(appid)][decision] >= int(observation_first_layer[decision][appid]))
            store_episode_1(observation_first_layer_copy, action_1)
        assert (np.sum(observation_first_layer, axis=1) <= params['container_limitation per node'] * 9).all()
        assert sum(sum(observation_first_layer)) == NUM_CONTAINERS
        # error_1 = 0
        # for node in range(3):
        #     for app in range(env.NUM_APPS):
        #         if observation_first_layer[node, :].sum() > params['container_limitation per node']*9 or observation_first_layer[node, app] > 9:
        #             error_1 += 1
        #     if observation_first_layer[node,1] + observation_first_layer[node,2] > 9:
        #         error_1 += 1
        # assert (error_1 == 0)

        """
        second layer
        """
        observation_second_layer_aggregation = np.empty([0, env.NUM_APPS], int)  # 9*20

        number_cont_second_layer = []

        for second_layer_index in range(nodes_per_group):

            rnd_array = observation_first_layer[second_layer_index].copy()
            total = rnd_array
            limit = (1 * 3)
            capicity = (8 * 3)  # 3
            s = Solver()
            # app sum == batch
            for i in range(7):
                s.add(z3.Sum(names['x' + str(i)]) == int(total[i]))
            # node capacity
            for node in range(3):
                s.add(z3.Sum([names['x' + str(i)][node] for i in range(7)]) <= int(capicity))
            # >=0
            for i in range(7):
                for node in range(3):
                    s.add(names['x' + str(i)][node] >= 0)
            # per app spread
            for i in range(7):
                for node in range(3):
                    s.add(names['x' + str(i)][node] <= limit)
            # App1 and App2 not exist
            for node in range(3):
                s.add(names['x' + str(1)][node] + names['x' + str(2)][node] <= limit)


            source_batch_second, index_data = batch_data_sub(rnd_array)
            observation_second_layer = np.zeros([nodes_per_group, env.NUM_APPS], int)
            NUM_CONTAINERS_second = sum(source_batch_second)
            number_cont_second_layer.append(NUM_CONTAINERS_second)

            for inter_episode_index in range(NUM_CONTAINERS_second):
                appid = index_data[inter_episode_index]
                observation_second_layer_copy, mapping_index = handle_constraint(observation_second_layer, appid)
                assert len(mapping_index) > 0

                source_batch_second[appid] -= 1
                # observation_second_layer_copy = observation_second_layer.copy()
                observation_second_layer_copy[:, appid] += 1
                observation_second_layer_copy = np.append(observation_second_layer_copy,
                                                          observation_second_layer_copy > 3 * 2, axis=1)
                observation_second_layer_copy = np.append(observation_second_layer_copy,
                                                          observation_second_layer_copy.sum(axis=1).reshape(
                                                              nodes_per_group, 1), axis=1)
                observation_second_layer_copy = np.array(observation_second_layer_copy).reshape(1, -1)
                observation_second_layer_copy = np.append(observation_second_layer_copy, appid).reshape(1, -1)
                observation_second_layer_copy = np.append(observation_second_layer_copy,
                                                          np.array(source_batch_second)).reshape(1, -1)

                action_2, prob_weights = RL_2.choose_action(observation_second_layer_copy.copy())
                decision = mapping_index[action_2]
                observation_second_layer[decision, appid] += 1
                s.add(names['x' + str(appid)][decision] >= int(observation_second_layer[decision][appid]))
                store_episode_2(observation_second_layer_copy, action_2)

            observation_second_layer_aggregation = np.append(observation_second_layer_aggregation,
                                                             observation_second_layer, 0)
            assert (np.sum(observation_second_layer, axis=1) <= params['container_limitation per node'] * 3).all()
            assert sum(sum(observation_second_layer)) == NUM_CONTAINERS_second
            # error_1 = 0
            # for node in range(3):
            #     for app in range(env.NUM_APPS):
            #         if observation_second_layer[node, :].sum() > params['container_limitation per node'] * 3 or \
            #                 observation_second_layer[node, app] > 3:
            #             error_1 += 1
            #     if observation_second_layer[node, 1] + observation_second_layer[node, 2] > 3:
            #         error_1 += 1
            # assert (error_1 == 0)

        """
        third layer
        """
        observation_third_layer_aggregation = np.empty([0, env.NUM_APPS], int)  # 9*20
        number_cont_third_layer = []

        for third_layer_index in range(nodes_per_group * nodes_per_group):
            rnd_array = observation_second_layer_aggregation[third_layer_index].copy()

            total = rnd_array
            limit = (1 * 1)
            capicity = 8
            s = Solver()
            # app sum == batch
            for i in range(7):
                s.add(z3.Sum(names['x' + str(i)]) == int(total[i]))
            # node capacity
            for node in range(3):
                s.add(z3.Sum([names['x' + str(i)][node] for i in range(7)]) <= int(capicity))
            # >=0
            for i in range(7):
                for node in range(3):
                    s.add(names['x' + str(i)][node] >= 0)
            # per app spread
            for i in range(7):
                for node in range(3):
                    s.add(names['x' + str(i)][node] <= limit)
            # App1 and App2 not exist
            for node in range(3):
                s.add(names['x' + str(1)][node] + names['x' + str(2)][node] <= limit)

            source_batch_third, index_data = batch_data_sub(rnd_array)
            observation_third_layer = np.zeros([nodes_per_group, env.NUM_APPS], int)
            NUM_CONTAINERS_third = sum(source_batch_third)
            number_cont_third_layer.append(NUM_CONTAINERS_third)

            for inter_episode_index in range(NUM_CONTAINERS_third):
                appid = index_data[inter_episode_index]
                observation_third_layer_copy, mapping_index = handle_constraint(observation_third_layer, appid)
                assert len(mapping_index) > 0
                source_batch_third[appid] -= 1
                # observation_third_layer_copy = observation_third_layer.copy()
                observation_third_layer_copy[:, appid] += 1

                observation_third_layer_copy = np.append(observation_third_layer_copy,
                                                         observation_third_layer_copy > 1 * 2, axis=1)
                observation_third_layer_copy = np.append(observation_third_layer_copy,
                                                         observation_third_layer_copy.sum(axis=1).reshape(
                                                             nodes_per_group, 1), axis=1)
                observation_third_layer_copy = np.array(observation_third_layer_copy).reshape(1, -1)
                observation_third_layer_copy = np.append(observation_third_layer_copy, appid).reshape(1, -1)
                observation_third_layer_copy = np.append(observation_third_layer_copy,
                                                         np.array(source_batch_third)).reshape(1, -1)

                action_3, prob_weights = RL_3.choose_action(observation_third_layer_copy.copy())
                decision = mapping_index[action_3]
                observation_third_layer[decision, appid] += 1
                s.add(names['x' + str(appid)][decision] >= int(observation_third_layer[decision][appid]))
                store_episode_3(observation_third_layer_copy, action_3)

            observation_third_layer_aggregation = np.append(observation_third_layer_aggregation,
                                                            observation_third_layer, 0)
            assert (np.sum(observation_third_layer, axis=1) <= params['container_limitation per node'] * 1).all()
            assert sum(sum(observation_third_layer)) == NUM_CONTAINERS_third
            # error_1 = 0
            # for node in range(3):
            #     for app in range(env.NUM_APPS):
            #         if observation_third_layer[node, :].sum() > params['container_limitation per node'] * 1 or \
            #                 observation_third_layer[node, app] > 1:
            #             error_1 += 1
            #     if observation_third_layer[node, 1] + observation_third_layer[node, 2] > 1:
            #         error_1 += 1
            # assert (error_1 == 0)
        time_al_end = time.time()
        time_al_acc += time_al_end - time_al_start
        """
        After an entire allocation, calculate total throughput, reward
        """
        env.state = observation_third_layer_aggregation.copy()
        assert sum(sum(env.state)) == NUM_CONTAINERS
        assert (env.state.sum(0) == source_batch_).all()
        tput_state = env.state
        tput_breakdown = sim.predict(tput_state.reshape(-1, env.NUM_APPS))
        tput = (tput_breakdown * tput_state).sum() / NUM_CONTAINERS
        reward_ratio = (tput - 0)

        state = env.state
        # These three are not actually used in training, just for logging
        list_check_per_app = (env.state > 1).sum() + max((env.state - 1).max(), 0)
        list_check_sum = sum(env.state.sum(1) > params['container_limitation per node']) + max(
            max(env.state.sum(1) - params['container_limitation per node']), 0)
        list_check_coex = sum((env.state[:, 1] > 0) * (env.state[:, 2] > 0))

        # list_check = list_check_sum + list_check_coex + list_check_per_app
        list_check = 0
        error = 0
        for node in range(NUM_NODES):
            for app in range(env.NUM_APPS):
                if env.state[node, app] > 1 or (app == 1 and env.state[node, 2] > 0) or (app == 2 and env.state[node, 1] > 0):
                    error += env.state[node, app]
        assert error==0

        # container limitation & deployment spread
        for node in range(NUM_NODES):
            for app in range(env.NUM_APPS):
                if env.state[node, :].sum() > params['container_limitation per node'] :#or env.state[node, app] > 1:
                    list_check += env.state[node, app]
        # hardware affinity & increamental deployment
        for app in range(7):
            node_now = np.where(env.state[:, app] > 0)[0]
            for node_ in node_now:
                if node_ not in app_node_set[app]:
                    list_check += env.state[node_, app]

        list_check_ratio = list_check / NUM_CONTAINERS

        safety_episode_1 = [list_check_ratio * 1.0] * len(observation_episode_1)
        reward_episode_1 = [reward_ratio * 1.0] * len(observation_episode_1)

        safety_episode_2 = [list_check_ratio * 1.0] * len(observation_episode_2)
        reward_episode_2 = [reward_ratio * 1.0] * len(observation_episode_2)

        safety_episode_3 = [list_check_ratio * 1.0] * len(observation_episode_3)
        reward_episode_3 = [reward_ratio * 1.0] * len(observation_episode_3)

        RL_1.store_tput_per_episode(tput, epoch_i, list_check, list_check_per_app, list_check_coex, list_check_sum)
        RL_2.store_tput_per_episode(tput, epoch_i, list_check, [], [], [])
        RL_3.store_tput_per_episode(tput, epoch_i, list_check, [], [], [])

        RL_1.store_training_samples_per_episode(observation_episode_1, action_episode_1, reward_episode_1,
                                                safety_episode_1)
        RL_2.store_training_samples_per_episode(observation_episode_2, action_episode_2, reward_episode_2,
                                                safety_episode_2)
        RL_3.store_training_samples_per_episode(observation_episode_3, action_episode_3, reward_episode_3,
                                                safety_episode_3)

        """
        check_tput_quality(tput)
        """
        if names['lowest_vio_' + str(tput_origimal_class)] > list_check:
            names['lowest_vio_' + str(tput_origimal_class)] = list_check
            names['observation_optimal_1_vio_' + str(tput_origimal_class)], names[
                'action_optimal_1_vio_' + str(tput_origimal_class)], names[
                'observation_optimal_2_vio_' + str(tput_origimal_class)], names[
                'action_optimal_2_vio_' + str(tput_origimal_class)], names[
                'number_optimal_vio_' + str(tput_origimal_class)], names[
                'safety_optimal_vio_1_' + str(tput_origimal_class)], names[
                'safety_optimal_vio_2_' + str(tput_origimal_class)], names[
                'safety_optimal_vio_3_' + str(tput_origimal_class)] = [], [], [], [], [], [], [], []
            names['observation_optimal_3_vio_' + str(tput_origimal_class)], names[
                'action_optimal_3_vio_' + str(tput_origimal_class)] = [], []
            names['reward_optimal_vio_' + str(tput_origimal_class)] = []
            names['observation_optimal_1_vio_' + str(tput_origimal_class)].extend(observation_episode_1)
            names['action_optimal_1_vio_' + str(tput_origimal_class)].extend(action_episode_1)
            names['observation_optimal_2_vio_' + str(tput_origimal_class)].extend(observation_episode_2)
            names['action_optimal_2_vio_' + str(tput_origimal_class)].extend(action_episode_2)
            names['observation_optimal_3_vio_' + str(tput_origimal_class)].extend(observation_episode_3)
            names['action_optimal_3_vio_' + str(tput_origimal_class)].extend(action_episode_3)
            names['number_optimal_vio_' + str(tput_origimal_class)].append(NUM_CONTAINERS)
            names['safety_optimal_vio_1_' + str(tput_origimal_class)].extend(safety_episode_1)
            names['safety_optimal_vio_2_' + str(tput_origimal_class)].extend(safety_episode_2)
            names['safety_optimal_vio_3_' + str(tput_origimal_class)].extend(safety_episode_3)
            names['reward_optimal_vio_' + str(tput_origimal_class)].extend(reward_episode_1)

            names['optimal_range_vio_' + str(tput_origimal_class)] = 1.1
        elif names['lowest_vio_' + str(tput_origimal_class)] >= list_check / names[
            'optimal_range_vio_' + str(tput_origimal_class)]:
            names['observation_optimal_1_vio_' + str(tput_origimal_class)].extend(observation_episode_1)
            names['action_optimal_1_vio_' + str(tput_origimal_class)].extend(action_episode_1)
            names['observation_optimal_2_vio_' + str(tput_origimal_class)].extend(observation_episode_2)
            names['action_optimal_2_vio_' + str(tput_origimal_class)].extend(action_episode_2)
            names['observation_optimal_3_vio_' + str(tput_origimal_class)].extend(observation_episode_3)
            names['action_optimal_3_vio_' + str(tput_origimal_class)].extend(action_episode_3)
            names['number_optimal_vio_' + str(tput_origimal_class)].append(NUM_CONTAINERS)
            names['safety_optimal_vio_1_' + str(tput_origimal_class)].extend(safety_episode_1)
            names['safety_optimal_vio_2_' + str(tput_origimal_class)].extend(safety_episode_2)
            names['safety_optimal_vio_3_' + str(tput_origimal_class)].extend(safety_episode_3)
            names['reward_optimal_vio_' + str(tput_origimal_class)].extend(reward_episode_1)

        # if list_check_ratio <= safety_requirement*0.5:
        if list_check_ratio <= safety_requirement:
            if names['highest_tput_' + str(tput_origimal_class)] < tput:
                names['highest_tput_' + str(tput_origimal_class)] = tput

                names['observation_optimal_1_' + str(tput_origimal_class)], names[
                    'action_optimal_1_' + str(tput_origimal_class)], names[
                    'observation_optimal_2_' + str(tput_origimal_class)], names[
                    'action_optimal_2_' + str(tput_origimal_class)], \
                names['reward_optimal_1_' + str(tput_origimal_class)], names[
                    'reward_optimal_2_' + str(tput_origimal_class)], names[
                    'reward_optimal_3_' + str(tput_origimal_class)], \
                names['number_optimal_' + str(tput_origimal_class)], \
                names['safety_optimal_1_' + str(tput_origimal_class)], names[
                    'safety_optimal_2_' + str(tput_origimal_class)], names[
                    'safety_optimal_3_' + str(tput_origimal_class)] \
                    = [], [], [], [], [], [], [], [], [], [], []
                names['observation_optimal_3_' + str(tput_origimal_class)], names[
                    'action_optimal_3_' + str(tput_origimal_class)] = [], []

                names['observation_optimal_1_' + str(tput_origimal_class)].extend(observation_episode_1)
                names['action_optimal_1_' + str(tput_origimal_class)].extend(action_episode_1)
                names['observation_optimal_2_' + str(tput_origimal_class)].extend(observation_episode_2)
                names['action_optimal_2_' + str(tput_origimal_class)].extend(action_episode_2)
                names['observation_optimal_3_' + str(tput_origimal_class)].extend(observation_episode_3)
                names['action_optimal_3_' + str(tput_origimal_class)].extend(action_episode_3)

                names['number_optimal_' + str(tput_origimal_class)].append(NUM_CONTAINERS)

                names['safety_optimal_1_' + str(tput_origimal_class)].extend(safety_episode_1)
                names['safety_optimal_2_' + str(tput_origimal_class)].extend(safety_episode_2)
                names['safety_optimal_3_' + str(tput_origimal_class)].extend(safety_episode_3)
                names['reward_optimal_1_' + str(tput_origimal_class)].extend(reward_episode_1)
                names['reward_optimal_2_' + str(tput_origimal_class)].extend(reward_episode_2)
                names['reward_optimal_3_' + str(tput_origimal_class)].extend(reward_episode_3)

                names['optimal_range_' + str(tput_origimal_class)] = 1.05

            elif names['highest_tput_' + str(tput_origimal_class)] < tput * names[
                'optimal_range_' + str(tput_origimal_class)]:
                names['observation_optimal_1_' + str(tput_origimal_class)].extend(observation_episode_1)
                names['action_optimal_1_' + str(tput_origimal_class)].extend(action_episode_1)
                names['observation_optimal_2_' + str(tput_origimal_class)].extend(observation_episode_2)
                names['action_optimal_2_' + str(tput_origimal_class)].extend(action_episode_2)
                names['observation_optimal_3_' + str(tput_origimal_class)].extend(observation_episode_3)
                names['action_optimal_3_' + str(tput_origimal_class)].extend(action_episode_3)

                names['number_optimal_' + str(tput_origimal_class)].append(NUM_CONTAINERS)

                names['safety_optimal_1_' + str(tput_origimal_class)].extend(safety_episode_1)
                names['safety_optimal_2_' + str(tput_origimal_class)].extend(safety_episode_2)
                names['safety_optimal_3_' + str(tput_origimal_class)].extend(safety_episode_3)
                names['reward_optimal_1_' + str(tput_origimal_class)].extend(reward_episode_1)
                names['reward_optimal_2_' + str(tput_origimal_class)].extend(reward_episode_2)
                names['reward_optimal_3_' + str(tput_origimal_class)].extend(reward_episode_3)

        observation_episode_1, action_episode_1, reward_episode_1, safety_episode_1 = [], [], [], []
        observation_episode_2, action_episode_2, reward_episode_2, safety_episode_2 = [], [], [], []
        observation_episode_3, action_episode_3, reward_episode_3, safety_episode_3 = [], [], [], []

        """
        Each batch, RL.learn()
        """
        if (epoch_i % batch_size == 0) & (epoch_i > 1):
            for replay_class in range(0, 1):

                number_optimal = names['number_optimal_' + str(replay_class)]

                reward_optimal_1 = names['reward_optimal_1_' + str(replay_class)]
                reward_optimal_2 = names['reward_optimal_2_' + str(replay_class)]
                reward_optimal_3 = names['reward_optimal_3_' + str(replay_class)]
                safety_optimal_1 = names['safety_optimal_1_' + str(replay_class)]
                safety_optimal_2 = names['safety_optimal_2_' + str(replay_class)]
                safety_optimal_3 = names['safety_optimal_3_' + str(replay_class)]

                observation_optimal_1 = names['observation_optimal_1_' + str(replay_class)]
                action_optimal_1 = names['action_optimal_1_' + str(replay_class)]
                observation_optimal_2 = names['observation_optimal_2_' + str(replay_class)]
                action_optimal_2 = names['action_optimal_2_' + str(replay_class)]
                observation_optimal_3 = names['observation_optimal_3_' + str(replay_class)]
                action_optimal_3 = names['action_optimal_3_' + str(replay_class)]

                buffer_size = int(len(number_optimal))

                if buffer_size < replay_size:
                    # TODO: if layers changes, training_times_per_episode should be modified
                    RL_1.ep_obs.extend(observation_optimal_1)
                    RL_1.ep_as.extend(action_optimal_1)
                    RL_1.ep_rs.extend(reward_optimal_1)
                    RL_1.ep_ss.extend(safety_optimal_1)

                    RL_2.ep_obs.extend(observation_optimal_2)
                    RL_2.ep_as.extend(action_optimal_2)
                    RL_2.ep_rs.extend(reward_optimal_2)
                    RL_2.ep_ss.extend(safety_optimal_2)

                    RL_3.ep_obs.extend(observation_optimal_3)
                    RL_3.ep_as.extend(action_optimal_3)
                    RL_3.ep_rs.extend(reward_optimal_3)
                    RL_3.ep_ss.extend(safety_optimal_3)

                else:
                    replay_index = np.random.choice(range(buffer_size), size=replay_size, replace=False)
                    for replay_id in range(replay_size):
                        replace_start = replay_index[replay_id]
                        start_location = sum(number_optimal[:replace_start])
                        stop_location = sum(number_optimal[:replace_start + 1])
                        RL_1.ep_obs.extend(observation_optimal_1[start_location: stop_location])
                        RL_1.ep_as.extend(action_optimal_1[start_location: stop_location])
                        RL_1.ep_rs.extend(reward_optimal_1[start_location: stop_location])
                        RL_1.ep_ss.extend(safety_optimal_1[start_location: stop_location])

                        RL_2.ep_obs.extend(observation_optimal_2[start_location: stop_location])
                        RL_2.ep_as.extend(action_optimal_2[start_location: stop_location])
                        RL_2.ep_rs.extend(reward_optimal_2[start_location: stop_location])
                        RL_2.ep_ss.extend(safety_optimal_2[start_location: stop_location])

                        RL_3.ep_obs.extend(observation_optimal_3[start_location: stop_location])
                        RL_3.ep_as.extend(action_optimal_3[start_location: stop_location])
                        RL_3.ep_rs.extend(reward_optimal_3[start_location: stop_location])
                        RL_3.ep_ss.extend(safety_optimal_3[start_location: stop_location])

            if not RL_1.start_cpo:
                for replay_class in range(0, 1):
                    number_optimal = names['number_optimal_vio_' + str(replay_class)]
                    safety_optimal_1 = names['safety_optimal_vio_1_' + str(replay_class)]
                    safety_optimal_2 = names['safety_optimal_vio_2_' + str(replay_class)]
                    safety_optimal_3 = names['safety_optimal_vio_3_' + str(replay_class)]
                    reward_optimal = names['reward_optimal_vio_' + str(replay_class)]

                    observation_optimal_1 = names['observation_optimal_1_vio_' + str(replay_class)]
                    action_optimal_1 = names['action_optimal_1_vio_' + str(replay_class)]
                    observation_optimal_2 = names['observation_optimal_2_vio_' + str(replay_class)]
                    action_optimal_2 = names['action_optimal_2_vio_' + str(replay_class)]
                    observation_optimal_3 = names['observation_optimal_3_vio_' + str(replay_class)]
                    action_optimal_3 = names['action_optimal_3_vio_' + str(replay_class)]

                    buffer_size = int(len(number_optimal))

                    if buffer_size < replay_size:
                        # TODO: if layers changes, training_times_per_episode should be modified
                        RL_1.ep_obs.extend(observation_optimal_1)
                        RL_1.ep_as.extend(action_optimal_1)
                        RL_1.ep_ss.extend(safety_optimal_1)
                        RL_1.ep_rs.extend(reward_optimal)

                        RL_2.ep_obs.extend(observation_optimal_2)
                        RL_2.ep_as.extend(action_optimal_2)
                        RL_2.ep_rs.extend(reward_optimal)
                        RL_2.ep_ss.extend(safety_optimal_2)

                        RL_3.ep_obs.extend(observation_optimal_3)
                        RL_3.ep_as.extend(action_optimal_3)
                        RL_3.ep_rs.extend(reward_optimal)
                        RL_3.ep_ss.extend(safety_optimal_3)

                    else:
                        replay_index = np.random.choice(range(buffer_size), size=replay_size, replace=False)
                        for replay_id in range(replay_size):
                            replace_start = replay_index[replay_id]
                            start_location = sum(number_optimal[:replace_start])
                            stop_location = sum(number_optimal[:replace_start + 1])
                            RL_1.ep_obs.extend(observation_optimal_1[start_location: stop_location])
                            RL_1.ep_as.extend(action_optimal_1[start_location: stop_location])
                            RL_1.ep_rs.extend(reward_optimal[start_location: stop_location])
                            RL_1.ep_ss.extend(safety_optimal_1[start_location: stop_location])

                            RL_2.ep_obs.extend(observation_optimal_2[start_location: stop_location])
                            RL_2.ep_as.extend(action_optimal_2[start_location: stop_location])
                            RL_2.ep_rs.extend(reward_optimal[start_location: stop_location])
                            RL_2.ep_ss.extend(safety_optimal_2[start_location: stop_location])

                            RL_3.ep_obs.extend(observation_optimal_3[start_location: stop_location])
                            RL_3.ep_as.extend(action_optimal_3[start_location: stop_location])
                            RL_3.ep_rs.extend(reward_optimal[start_location: stop_location])
                            RL_3.ep_ss.extend(safety_optimal_3[start_location: stop_location])

            time_s = time.time()
            RL_1.learn(epoch_i, thre_entropy, Ifprint=True)
            RL_2.learn(epoch_i, thre_entropy)
            optim_case = RL_3.learn(epoch_i, thre_entropy)
            time_e = time.time()
            print("learning time epoch_i:", epoch_i, time_e - time_s)
            print("End2End time epoch_i", epoch_i, time_ep_acc)
            print("Allocate time epoch_i", epoch_i, time_al_acc)
            time_al_acc = 0.0
            time_ep_acc = 0.0
        """
        checkpoint, per 1000 episodes
        """
        if (epoch_i % 3000 == 0) & (epoch_i > 1):

            RL_1.save_session(ckpt_path_1)
            RL_2.save_session(ckpt_path_2)
            RL_3.save_session(ckpt_path_3)
            np.savez(np_path, tputs=np.array(RL_1.tput_persisit), candidate=np.array(RL_1.episode),
                     vi_perapp=np.array(RL_1.ss_perapp_persisit), vi_coex=np.array(RL_1.ss_coex_persisit),
                     vi_sum=np.array(RL_1.ss_sum_persisit))

            """
            optimal range adaptively change
            """
            for class_replay in range(0, 1):
                number_optimal = names['number_optimal_' + str(class_replay)]
                count_size = int(len(number_optimal))

                if (count_size > 300):
                    names['optimal_range_' + str(class_replay)] *= 0.99
                    names['optimal_range_' + str(class_replay)] = max(names['optimal_range_' + str(class_replay)], 1.01)
                    start_location = sum(
                        names['number_optimal_' + str(class_replay)][:-50]) * training_times_per_episode
                    names['observation_optimal_1_' + str(class_replay)] = names['observation_optimal_1_' + str(
                        class_replay)][start_location:]
                    names['action_optimal_1_' + str(class_replay)] = names['action_optimal_1_' + str(class_replay)][
                                                                     start_location:]
                    names['observation_optimal_2_' + str(class_replay)] = names['observation_optimal_2_' + str(
                        class_replay)][start_location:]
                    names['action_optimal_2_' + str(class_replay)] = names['action_optimal_2_' + str(class_replay)][
                                                                     start_location:]
                    names['observation_optimal_3_' + str(class_replay)] = names['observation_optimal_3_' + str(
                        class_replay)][start_location:]
                    names['action_optimal_3_' + str(class_replay)] = names['action_optimal_3_' + str(class_replay)][
                                                                     start_location:]
                    names['number_optimal_' + str(class_replay)] = names['number_optimal_' + str(class_replay)][-50:]
                    names['safety_optimal_1_' + str(class_replay)] = names['safety_optimal_1_' + str(class_replay)][
                                                                     start_location:]
                    names['safety_optimal_2_' + str(class_replay)] = names['safety_optimal_2_' + str(class_replay)][
                                                                     start_location:]
                    names['safety_optimal_3_' + str(class_replay)] = names['safety_optimal_3_' + str(class_replay)][
                                                                     start_location:]
                    names['reward_optimal_1_' + str(class_replay)] = names['reward_optimal_1_' + str(class_replay)][
                                                                     start_location:]
                    names['reward_optimal_2_' + str(class_replay)] = names['reward_optimal_2_' + str(class_replay)][
                                                                     start_location:]
                    names['reward_optimal_3_' + str(class_replay)] = names['reward_optimal_3_' + str(class_replay)][
                                                                     start_location:]

                print("optimal_range:", names['optimal_range_' + str(class_replay)])

            thre_entropy *= 0.5
            thre_entropy = max(thre_entropy, 0.0001)

        epoch_i += 1

        time_ep_end = time.time()
        time_ep_acc += time_ep_end - time_ep_start

        if epoch_i > 10000:
            batch_size = 100


def batch_data(NUM_CONTAINERS, NUM_NODES):
    # npzfile = np.load("./data/batch_set_cpo_27node_" + str(100) + '.npz')
    # # batch_set = npzfile['batch_set']
    # batch_set = npzfile['arr_0']
    rnd_array = [21,	13,	14,	9,	16,	15,	12]
    index_data = []
    for i in range(7):
        index_data.extend([i] * rnd_array[i])

    print(hyper_parameter['batch_C_numbers'])
    print(rnd_array)
    return rnd_array, index_data


def batch_data_sub(rnd_array):
    rnd_array = rnd_array.copy()
    index_data = []
    for i in range(7):
        index_data.extend([i] * int(rnd_array[i]))

    return rnd_array, index_data


def make_path(dirname):
    if not os.path.exists("./checkpoint/" + dirname):
        os.mkdir("./checkpoint/" + dirname)
        print("Directory ", "./checkpoint/" + dirname, " Created ")
    else:
        print("Directory ", "./checkpoint/" + dirname, " already exists")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_choice', type=int, default=0)
    parser.add_argument('--clip_eps', type=float, default=0.2)
    parser.add_argument('--epochs', type=int, default=50000)
    parser.add_argument('--batch_size_tunning', type=int, default=20)
    parser.add_argument('--rp_size', type=int, default=100)
    parser.add_argument('--safety_requirement', type=float, default=0.05)
    parser.add_argument('--recover', action='store_true')
    # parser.add_argument('--not_norm_proj', action='store_true')
    parser.add_argument('--lr', type=float, default=0.001)
    args = parser.parse_args()

    hyper_parameter['batch_C_numbers'] = args.batch_choice
    params['path'] = "pppo1006_27_" + str(hyper_parameter['batch_C_numbers'])

    params['batch_size'] = args.batch_size_tunning
    # params['epochs'] = params['epochs'] * (params['batch_size'] / 50)
    params['epochs'] = args.epochs

    params['clip_eps'] = args.clip_eps
    params['safety_requirement'] = args.safety_requirement
    params['recover'] = args.recover
    # params['not_norm_proj'] = args.not_norm_proj
    params['learning rate'] = args.lr

    make_path(params['path'])
    train(params)


def plot():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_choice', type=int)
    args = parser.parse_args()
    choice = args.batch_choice

    plot_meta("pppo1006_27_" + str(choice), "Throughput PPPO")
    # plot_meta("60C_basic_limit10_128128", "Throughput Policy Gradient", plot_violation=False)
    plt.legend(loc='best')
    plt.xlabel("episode")
    plt.ylabel("violation")
    # plt.title(params["path"])
    # plt.title("# of containers: "+ str(params['NUM_CONTAINERS_start']) + " - " +str(params['NUM_CONTAINERS_start'] + 10))
    # plt.title("Constraint: app#1,2 no co-exist")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    # plt.savefig("./Constraint: sum(node) and app#1,2 no co-exist" + ".pdf", format='pdf', dpi=2400, transparent=True)


def plot_meta(name, label_name, plot_violation=True):
    np_path = "./checkpoint/" + name + "/optimal_file_name.npz"
    npzfile = np.load(np_path)

    tput = npzfile['tputs']  # [0:50000] * 1.05
    # vio = npzfile['vio'][0:50000]
    epoch = npzfile['candidate']
    epoch = epoch  # [0:50000]
    epoch = epoch
    window_size = 50
    lenth = len(tput)

    tput_smooth = np.convolve(tput, np.ones(window_size, dtype=int), 'valid')
    epoch_smooth = np.convolve(epoch, np.ones(window_size, dtype=int), 'valid')

    # tput = npzfile['tputs']
    # epoch = npzfile['candidate']
    # window_size = 10
    # tput_smooth = np.convolve(tput, np.ones(window_size, dtype=int), 'valid')
    # epoch_smooth = np.convolve(epoch, np.ones(window_size, dtype=int), 'valid')

    plt.subplot(211)
    plt.plot(1.0 * epoch_smooth / window_size, 1.0 * tput_smooth / window_size, '.', label=label_name)
    plt.grid(True)
    # if plot_violation:
    #     vio = npzfile['vio']
    #     vio_smooth = np.convolve(vio, np.ones(window_size, dtype=int), 'valid')
    #     plt.plot(1.0 * epoch_smooth / window_size, 1.0 * vio_smooth / window_size, '.', label="Violation CPO")
    if plot_violation:
        plt.subplot(212)
        vio_coex = npzfile['vi_sum'] + npzfile['vi_coex'] + npzfile['vi_perapp']
        vio_coex_smooth = np.convolve(vio_coex, np.ones(window_size, dtype=int), 'valid')
        plt.plot(1.0 * epoch_smooth / window_size, 1.0 * vio_coex_smooth / window_size, '.', label="Violtion (sum)")

        # vio_coex = npzfile['vi_coex']
        # vio_coex_smooth = np.convolve(vio_coex, np.ones(window_size, dtype=int), 'valid')
        # plt.plot(1.0 * epoch_smooth / window_size, 1.0 * vio_coex_smooth / window_size, '.', label="Violtion (co-exist)")
        #
        # vio_perapp = npzfile['vi_perapp']
        # vio_perapp_smooth = np.convolve(vio_perapp, np.ones(window_size, dtype=int), 'valid')
        # plt.plot(1.0 * epoch_smooth / window_size, 1.0 * vio_perapp_smooth / window_size, '.', label="Violtion (co-exist)")


if __name__ == "__main__":
    main()
    # plot()