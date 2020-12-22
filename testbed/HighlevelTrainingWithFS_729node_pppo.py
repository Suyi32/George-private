import numpy as np
import time
import os
import sys
sys.path.append("/workspace/George-private")
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from testbedlib.highlevel_env_pppo import LraClusterEnv
from testbedlib.PolicyGradient_PCPO_PPO import PolicyGradient
import argparse
import matplotlib.pyplot as plt
from z3 import *
import z3

"""
'--batch_choice': 0, 1, 2, ``` 30
'--container_N': 1000, 2000, 3000
python3 HighlevelTrainingWithFS_729node_pppo.py --container_N 1000 --batch_choice 0
"""

hyper_parameter = {
        'batch_C_numbers': None,
        'container_N':None
}

params = {
        'batch_size': 20,
        'epochs': 5000,
        'path': "pppo_729_fromscratch_" + str(hyper_parameter['container_N']) + "_" + str(hyper_parameter['batch_C_numbers']),
        'path_recover': "729_single_" + str(hyper_parameter['batch_C_numbers']),
        'recover': False,
        'number of containers': 2100,
        'learning rate': 0.001,
        'nodes per group': 3,
        'number of nodes in the cluster': 27,  # 81
        'replay size': 50,
        'container_limitation per node': 27*8   # 81
}


def choose_action_external_knowledge(observation_new_list, app_index):
    # external: new action-chosen method with external knowledge

    observation_list = observation_new_list[0, 0:-7].reshape(params['nodes per group'], 7) #[0:27]

    #: first, we select the nodes with min No. of containers
    index_min_total = np.where(observation_list.sum(1) == observation_list.sum(1).min())
    observation_list_second = observation_list[index_min_total]

    #: second, we select the nodes with min No. of app_index
    index_min_appindex = np.where(observation_list_second[:,int(app_index)] == observation_list_second[:, int(app_index)].min())
    # node = index_min_total[0][index_min_appindex[0][0]]
    node = index_min_total[0][np.random.choice(index_min_appindex[0])]

    return node, []


def train(params):

    """
    parameters set
    """
    NUM_NODES = params['number of nodes in the cluster']
    node_limit_sum = 120
    node_limit_coex = 20
    NUM_APPS = 7

    batch_size = params['batch_size']
    ckpt_path_1 = "./checkpoint/" + params['path'] + "1/model.ckpt"
    ckpt_path_2 = "./checkpoint/" + params['path'] + "2/model.ckpt"
    ckpt_path_3 = "./checkpoint/" + params['path'] + "3/model.ckpt"
    make_path(params['path'] + "1")
    make_path(params['path'] + "2")
    make_path(params['path'] + "3")

    ckpt_path_recover_1 = "../results/cpo/newhypernode/" + params['path_recover'] + "1/model.ckpt"
    ckpt_path_recover_2 = "../results/cpo/newhypernode/" + params['path_recover'] + "2/model.ckpt"
    ckpt_path_recover_3 = "../results/cpo/newhypernode/" + params['path_recover'] + "3/model.ckpt"

    env = LraClusterEnv(num_nodes=NUM_NODES)

    np_path = "./checkpoint/" + params['path'] + "/optimal_file_name.npz"
    Recover = params['recover']
    nodes_per_group = int(params['nodes per group'])
    replay_size = params['replay size']
    training_times_per_episode = 1  # TODO: if layers changes, training_times_per_episode should be modified
    safety_requirement = 0.05#40
    ifUseExternal = False

    """
    Build Network
    """
    n_actions = nodes_per_group  #: 3 nodes per group
    n_features = int(n_actions * (env.NUM_APPS + 1 + env.NUM_APPS )+ 1 + env.NUM_APPS)  #: 3*9+1 = 28
    RL_1 = PolicyGradient(
        n_actions=n_actions,
        n_features=n_features,
        learning_rate=params['learning rate'],
        suffix='1b',
        safety_requirement=safety_requirement)

    RL_2 = PolicyGradient(
        n_actions=n_actions,
        n_features=n_features,
        learning_rate=params['learning rate'],
        suffix='2b',
        safety_requirement=safety_requirement)

    RL_3 = PolicyGradient(
        n_actions=n_actions,
        n_features=n_features,
        learning_rate=params['learning rate'],
        suffix='3b',
        safety_requirement=safety_requirement)

    # sim = Simulator()

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

    thre_entropy = 0.001
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

    def store_episode_1(observations, actions):
        observation_episode_1.append(observations)
        action_episode_1.append(actions)

    def store_episode_2(observations, actions):
        observation_episode_2.append(observations)
        action_episode_2.append(actions)

    def store_episode_3(observations, actions):
        observation_episode_3.append(observations)
        action_episode_3.append(actions)

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

    source_batch_a, index_data_a = batch_data()  # index_data = [0,1,2,0,1,2]

    while epoch_i < params['epochs']:
        if Recover:
            RL_1.restore_session(ckpt_path_recover_1)
            RL_2.restore_session(ckpt_path_recover_2)
            RL_3.restore_session(ckpt_path_recover_3)
            Recover = False

        tput_origimal_class = 0
        source_batch_ = source_batch_a.copy()
        index_data = index_data_a.copy()
        NUM_CONTAINERS = sum(source_batch_)
        observation = np.zeros([NUM_NODES, NUM_APPS]).copy()  # (9,9)
        source_batch = source_batch_.copy()

        """
        Episode
        """
        """
        first layer
        """
        total = source_batch
        limit = (1 * 9 * 27)
        capicity = (8 * 9 * 27)  # 3
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

            observation_first_layer_copy = np.append(observation_first_layer_copy, observation_first_layer_copy > 9 * node_limit_coex, axis=1)
            observation_first_layer_copy = np.append(observation_first_layer_copy, observation_first_layer_copy.sum(axis=1).reshape(nodes_per_group, 1), axis=1)
            # observation_first_layer_copy = np.append(observation_first_layer_copy, ((observation_first_layer_copy[:, 2] > 0) * (observation_first_layer_copy[:, 3] > 0)).reshape(nodes_per_group, 1), axis=1)
            observation_first_layer_copy = np.array(observation_first_layer_copy).reshape(1, -1)
            observation_first_layer_copy = np.append(observation_first_layer_copy, appid).reshape(1, -1)
            observation_first_layer_copy = np.append(observation_first_layer_copy, np.array(source_batch_first)).reshape(1, -1)
            if ifUseExternal:
                action_1 = inter_episode_index % 3
                prob_weights = []
            else:
                action_1, prob_weights = RL_1.choose_action(observation_first_layer_copy.copy())

            decision = mapping_index[action_1]
            observation_first_layer[decision, appid] += 1
            s.add(names['x' + str(appid)][decision] >= int(observation_first_layer[decision][appid]))

            store_episode_1(observation_first_layer_copy, action_1)
        assert (np.sum(observation_first_layer, axis=1) <= params['container_limitation per node'] * 9).all()
        assert sum(sum(observation_first_layer)) == NUM_CONTAINERS

        """
        second layer
        """
        observation_second_layer_aggregation = np.empty([0, env.NUM_APPS], int)  # 9*20

        number_cont_second_layer = []

        for second_layer_index in range(nodes_per_group):

            rnd_array = observation_first_layer[second_layer_index].copy()
            total = rnd_array
            limit = (1 * 3 *27)
            capicity = (8 * 3*27)  # 3
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

                observation_second_layer_copy = np.append(observation_second_layer_copy, observation_second_layer_copy > 3 * node_limit_coex, axis=1)
                observation_second_layer_copy = np.append(observation_second_layer_copy, observation_second_layer_copy.sum(axis=1).reshape(nodes_per_group, 1), axis=1)
                # observation_second_layer_copy = np.append(observation_second_layer_copy, ((observation_second_layer_copy[:, 2] > 0) * (observation_second_layer_copy[:, 3] > 0)).reshape(nodes_per_group, 1), axis=1)
                observation_second_layer_copy = np.array(observation_second_layer_copy).reshape(1, -1)
                observation_second_layer_copy = np.append(observation_second_layer_copy, appid).reshape(1, -1)
                observation_second_layer_copy = np.append(observation_second_layer_copy, np.array(source_batch_second)).reshape(1, -1)
                if ifUseExternal:
                    action_2 = inter_episode_index % 3
                    prob_weights = []
                else:
                    action_2, prob_weights = RL_2.choose_action(observation_second_layer_copy.copy())

                decision = mapping_index[action_2]
                observation_second_layer[decision, appid] += 1
                s.add(names['x' + str(appid)][decision] >= int(observation_second_layer[decision][appid]))
                store_episode_2(observation_second_layer_copy, action_2)
            assert (np.sum(observation_second_layer, axis=1) <= params['container_limitation per node'] * 3).all()
            assert sum(sum(observation_second_layer)) == NUM_CONTAINERS_second
            observation_second_layer_aggregation = np.append(observation_second_layer_aggregation, observation_second_layer, 0)

        """
        third layer
        """
        observation_third_layer_aggregation = np.empty([0, env.NUM_APPS], int)  # 9*20
        number_cont_third_layer = []

        for third_layer_index in range(nodes_per_group * nodes_per_group):

            rnd_array = observation_second_layer_aggregation[third_layer_index].copy()
            total = rnd_array
            limit = (1 * 1 *27)
            capicity = 8 *27
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

                observation_third_layer_copy = np.append(observation_third_layer_copy, observation_third_layer_copy > 1 * node_limit_coex, axis=1)
                observation_third_layer_copy = np.append(observation_third_layer_copy, observation_third_layer_copy.sum(axis=1).reshape(nodes_per_group, 1), axis=1)
                # observation_third_layer_copy = np.append(observation_third_layer_copy, ((observation_third_layer_copy[:, 2] > 0) * (observation_third_layer_copy[:, 3] > 0)).reshape(nodes_per_group, 1), axis=1)
                observation_third_layer_copy = np.array(observation_third_layer_copy).reshape(1, -1)
                observation_third_layer_copy = np.append(observation_third_layer_copy, appid).reshape(1, -1)
                observation_third_layer_copy = np.append(observation_third_layer_copy, np.array(source_batch_third)).reshape(1, -1)

                if ifUseExternal:
                    action_3 = inter_episode_index % 3
                    prob_weights = []
                else:

                    action_3, prob_weights = RL_3.choose_action(observation_third_layer_copy.copy())

                decision = mapping_index[action_3]
                observation_third_layer[decision, appid] += 1
                s.add(names['x' + str(appid)][decision] >= int(observation_third_layer[decision][appid]))

                store_episode_3(observation_third_layer_copy, action_3)

            observation_third_layer_aggregation = np.append(observation_third_layer_aggregation, observation_third_layer, 0)
            assert (np.sum(observation_third_layer, axis=1) <= params['container_limitation per node'] * 1).all()
            assert sum(sum(observation_third_layer)) == NUM_CONTAINERS_third
        """
        After an entire allocation, calculate total throughput, reward
        """
        env.state = observation_third_layer_aggregation.copy()
        assert sum(sum(env.state)) == NUM_CONTAINERS
        assert (env.state.sum(0) == source_batch_).all()

        total_tput, list_check_sum, list_check_coex, list_check_per_app, list_check = env.get_tput_total_env()

        tput = total_tput/NUM_CONTAINERS
        list_check = 1.0 * list_check / NUM_CONTAINERS
        reward_ratio = tput

        list_check_ratio = list_check

        list_check_layer_one = 0
        list_check_layer_one_ratio = list_check_layer_one

        safety_episode_1 = [list_check_ratio+ list_check_layer_one_ratio * 1.0] * len(observation_episode_1)
        reward_episode_1 = [reward_ratio * 1.0] * len(observation_episode_1)

        safety_episode_2 = [list_check_ratio * 1.0] * len(observation_episode_2)
        reward_episode_2 = [reward_ratio * 1.0] * len(observation_episode_2)

        safety_episode_3 = [list_check_ratio * 1.0] * len(observation_episode_3)
        reward_episode_3 = [reward_ratio * 1.0] * len(observation_episode_3)


        RL_1.store_tput_per_episode(tput, epoch_i, list_check+list_check_layer_one, list_check_per_app, list_check_coex, list_check_sum)
        RL_2.store_tput_per_episode(tput, epoch_i, list_check+list_check_layer_one, list_check_per_app, list_check_coex, list_check_sum)
        RL_3.store_tput_per_episode(tput, epoch_i, list_check+list_check_layer_one, list_check_per_app, list_check_coex, list_check_sum)


        RL_1.store_training_samples_per_episode(observation_episode_1, action_episode_1, reward_episode_1, safety_episode_1)
        RL_2.store_training_samples_per_episode(observation_episode_2, action_episode_2, reward_episode_2, safety_episode_2)
        RL_3.store_training_samples_per_episode(observation_episode_3, action_episode_3, reward_episode_3, safety_episode_3)

        """
        check_tput_quality(tput)
        """
        if list_check <= safety_requirement*0.8:
            if names['highest_tput_' + str(tput_origimal_class)] < tput:
                names['highest_tput_' + str(tput_origimal_class)] = tput

                names['observation_optimal_1_' + str(tput_origimal_class)], names['action_optimal_1_' + str(tput_origimal_class)], names['observation_optimal_2_' + str(tput_origimal_class)], names['action_optimal_2_' + str(tput_origimal_class)],\
                names['reward_optimal_1_' + str(tput_origimal_class)],names['reward_optimal_2_' + str(tput_origimal_class)],names['reward_optimal_3_' + str(tput_origimal_class)], \
                names['number_optimal_' + str(tput_origimal_class)],\
                names['safety_optimal_1_' + str(tput_origimal_class)],names['safety_optimal_2_' + str(tput_origimal_class)],names['safety_optimal_3_' + str(tput_origimal_class)]\
                    = [], [], [], [], [], [], [], [], [], [], []
                names['observation_optimal_3_' + str(tput_origimal_class)], names['action_optimal_3_' + str(tput_origimal_class)] = [], []

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

            elif names['highest_tput_' + str(tput_origimal_class)] < tput * names['optimal_range_' + str(tput_origimal_class)]:
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
            for replay_class in range(0,10):

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
                        stop_location = sum(number_optimal[:replace_start+1])
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
            #
            RL_1.learn(epoch_i, thre_entropy, Ifprint=True)
            RL_2.learn(epoch_i, thre_entropy)
            optim_case = RL_3.learn(epoch_i, thre_entropy)

        """
        checkpoint, per 1000 episodes
        """
        if (epoch_i % 200 == 0) & (epoch_i > 1):
            for class_replay in range(0,10):
                highest_value = names['highest_tput_' + str(class_replay)]
                print("\n epoch: %d, highest tput: %f" % (epoch_i, highest_value))

                # lowest_vio_ = names['lowest_vio_' + str(class_replay)]
                # print("\n epoch: %d, lowest_vio: %f" % (epoch_i, lowest_vio_))

            RL_1.save_session(ckpt_path_1)
            RL_2.save_session(ckpt_path_2)
            RL_3.save_session(ckpt_path_3)
            np.savez(np_path, tputs=np.array(RL_1.tput_persisit), candidate=np.array(RL_1.episode), vio_persis=np.array(RL_1.safe_persisit))
            print("epoch:", epoch_i, "mean(sum): ", np.mean(RL_1.sum_persisit), "mean(coex): ", np.mean(RL_1.coex_persisit))
            """
            optimal range adaptively change
            """
            for class_replay in range(0, 10):
                number_optimal = names['number_optimal_' + str(class_replay)]
                count_size = int(len(number_optimal))

                if (count_size > 100):
                    names['optimal_range_' + str(class_replay)] *= 0.99
                    names['optimal_range_' + str(class_replay)] = max(names['optimal_range_' + str(class_replay)], 1.01)

                    start_location = sum(names['number_optimal_' + str(class_replay)][:-10]) * training_times_per_episode

                    names['observation_optimal_1_' + str(class_replay)] = names['observation_optimal_1_' + str(class_replay)][start_location:]
                    names['action_optimal_1_' + str(class_replay)] = names['action_optimal_1_' + str(class_replay)][start_location:]

                    names['observation_optimal_2_' + str(class_replay)] = names['observation_optimal_2_' + str(class_replay)][start_location:]
                    names['action_optimal_2_' + str(class_replay)] = names['action_optimal_2_' + str(class_replay)][start_location:]

                    names['observation_optimal_3_' + str(class_replay)] = names['observation_optimal_3_' + str(class_replay)][start_location:]
                    names['action_optimal_3_' + str(class_replay)] = names['action_optimal_3_' + str(class_replay)][start_location:]

                    names['number_optimal_' + str(class_replay)] = names['number_optimal_' + str(class_replay)][-10:]

                    names['safety_optimal_1_' + str(class_replay)] = names['safety_optimal_1_' + str(class_replay)][start_location:]
                    names['safety_optimal_2_' + str(class_replay)] = names['safety_optimal_2_' + str(class_replay)][start_location:]
                    names['safety_optimal_3_' + str(class_replay)] = names['safety_optimal_3_' + str(class_replay)][start_location:]
                    names['reward_optimal_1_' + str(class_replay)] = names['reward_optimal_1_' + str(class_replay)][start_location:]
                    names['reward_optimal_2_' + str(class_replay)] = names['reward_optimal_2_' + str(class_replay)][start_location:]
                    names['reward_optimal_3_' + str(class_replay)] = names['reward_optimal_3_' + str(class_replay)][start_location:]

                print("optimal_range:", names['optimal_range_' + str(class_replay)])

            print(prob_weights)
            if optim_case > 0:
                thre_entropy *= 0.5
            thre_entropy = max(thre_entropy, 0.001)

        epoch_i += 1
        if epoch_i>30:
            ifUseExternal = False


def indices_to_one_hot(data, nb_classes):  #separate: embedding
    """Convert an iterable of indices to one-hot encoded labels."""
    targets = np.array(data).reshape(-1)
    return np.eye(nb_classes)[targets]


def batch_data():

    npzfile = np.load("data/batch_set_cpo_27node_" + str(hyper_parameter['container_N']) + '.npz')
    batch_set = npzfile['batch_set']
    rnd_array = batch_set[hyper_parameter['batch_C_numbers'], :]
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


def plot_meta(name,label_name):

    np_path = name + "/optimal_file_name.npz"
    npzfile = np.load(np_path)
    tput = npzfile['tputs'][30:]
    epoch = npzfile['time'][30:]
    window_size = 100
    tput_smooth = np.convolve(tput, np.ones(window_size, dtype=int), 'valid')
    epoch_smooth = np.convolve(epoch, np.ones(window_size, dtype=int), 'valid')
    plt.plot(1.0 * epoch_smooth / window_size, 1.0 * tput_smooth / window_size, '.', label=label_name)


def draw_graph_single(params):
    # plot_meta("../results/729/729_clustering_with_external-embedding", "Clustering")
    plot_meta('./checkpoint/'+params['path'], "Basic")
    plt.legend(loc = 4)
    plt.xlabel("time (minutes)")
    plt.ylabel("throughput")
    plt.title("1400 containers, 7 apps -> 729 nodes")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def make_path(dirname):

    if not os.path.exists("./checkpoint/" + dirname):
        os.mkdir("./checkpoint/"+ dirname)
        print("Directory ", "./checkpoint/" + dirname, " Created ")
    else:
        print("Directory ", "./checkpoint/" + dirname, " already exists")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_choice', type=int, default=0)
    parser.add_argument('--container_N', type=int, default=2000)
    args = parser.parse_args()
    hyper_parameter['batch_C_numbers'] = args.batch_choice
    hyper_parameter['container_N'] = args.container_N
    params['path'] = "729_single_" + str(hyper_parameter['batch_C_numbers'])
    make_path(params['path'])
    train(params)
    draw_graph_single(params)


if __name__ == "__main__":
    main()
