import numpy as np
import time
import os
import sys
sys.path.append("/workspace/atc/")

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from testbed.cluster_env import LraClusterEnv
from testbed.PolicyGradient_DNN import PolicyGradient
import matplotlib.pyplot as plt
import argparse
from testbed.simulator.simulator import Simulator
from z3 import *
import time

"""
fish
'NUM_CONTAINERS_start': 0, 10, 20, ``` 260
fish
# cd /home/lwangbm/cpo/testbed_3/
cd /home/ubuntu/cpo/testbed_3/
screen
python3 81c9node9app_Dynamic_large_differ_replay_separate_limit10_27_reverse.py --start_sample 0
"""
hyper_parameter = {
        'batch_C_numbers': None
}
params = {
        'batch_size': 50,
        'epochs': 11000,
        'path': "27node_1000",
        'recover': False,
        'learning rate': 0.01,
        'nodes per group': 3,
        'number of nodes in the cluster': 27,
        'replay size': 10,
        'number of containers': 100,
        'container_limitation per node': 8
}
names = locals()
for i in range(7):
    names['x' + str(i)] = z3.IntVector('x' + str(i), 27)


def train(params):
    time_epoch_set = []
    start_time = time.time()
    """
    parameters set
    """
    NUM_NODES = params['number of nodes in the cluster']
    NUM_CONTAINERS = params['number of containers']
    env = LraClusterEnv(num_nodes=NUM_NODES)
    batch_size = params['batch_size']
    ckpt_path_1 = "./checkpoint/" + params['path'] + "_1" + "/model.ckpt"
    ckpt_path_2 = "./checkpoint/" + params['path'] + "_2" + "/model.ckpt"
    ckpt_path_3 = "./checkpoint/" + params['path'] + "_3" + "/model.ckpt"
    np_path = "./checkpoint/" + params['path'] + "/optimal_file_name.npz"
    Recover = params['recover']
    nodes_per_group = int(params['nodes per group'])
    replay_size = params['replay size']
    training_times_per_episode = 1
    UseExperienceReplay = False


    """
    Build Network
    """
    n_actions = nodes_per_group  #: 3 nodes per group
    n_features = int(n_actions * env.NUM_APPS + 1 + env.NUM_APPS)  #: 3*7+1+7 = 29
    RL_1 = PolicyGradient(
        n_actions=n_actions,
        n_features=n_features,
        learning_rate=params['learning rate'],
        suffix=str(params['number of containers']) + '1')

    RL_2 = PolicyGradient(
        n_actions=n_actions,
        n_features=n_features,
        learning_rate=params['learning rate'],
        suffix=str(params['number of containers']) + '2')

    RL_3 = PolicyGradient(
        n_actions=n_actions,
        n_features=n_features,
        learning_rate=params['learning rate'],
        suffix=str(params['number of containers']) + '3')
    sim = Simulator()
    """
    Training
    """
    start_time = time.time()
    global_start_time = start_time

    observation_episode_1, action_episode_1, reward_episode_1 = [], [], []
    observation_episode_2, action_episode_2, reward_episode_2 = [], [], []
    observation_episode_3, action_episode_3, reward_episode_3 = [], [], []


    epoch_i = 0
    entropy_weight = 0.1
    for i in range(0,1):
        names['highest_tput_' + str(i)] = 0.1
        names['observation_optimal_1_' + str(i)] = []
        names['action_optimal_1_' + str(i)] = []
        names['reward_optimal_1_' + str(i)] = []
        names['number_optimal_' + str(i)] = []
        names['optimal_range_' + str(i)] = 1.2

    for i in range(0,1):
        names['observation_optimal_2_' + str(i)] = []
        names['action_optimal_2_' + str(i)] = []
        names['reward_optimal_2_' + str(i)] = []

    for i in range(0,1):
        names['observation_optimal_3_' + str(i)] = []
        names['action_optimal_3_' + str(i)] = []
        names['reward_optimal_3_' + str(i)] = []

    def store_episode_1(observations, actions):
        observation_episode_1.append(observations)
        action_episode_1.append(actions)

    def store_episode_2(observations, actions):
        observation_episode_2.append(observations)
        action_episode_2.append(actions)

    def store_episode_3(observations, actions):
        observation_episode_3.append(observations)
        action_episode_3.append(actions)

    while epoch_i < params['epochs']:


        tput_origimal_class = 0
        source_batch_, index_data = batch_data(NUM_CONTAINERS, env.NUM_APPS)  # index_data = [0,1,2,0,1,2]
        observation = env.reset().copy()  # (9,9)
        source_batch = source_batch_.copy()
        source_batch_cpoy = source_batch.copy()

        total = source_batch
        # observation = observation_original.copy()
        limit = (1 - observation)
        capicity = (params['container_limitation per node'] - observation.sum(1)).reshape(-1)  # 27
        s = Solver()
        # app sum == batch

        for i in range(7):
            s.add(z3.Sum(names['x' + str(i)]) == int(total[i]))

        # node capacity
        for node in range(27):
            s.add(z3.Sum([names['x' + str(i)][node] for i in range(7)]) <= int(capicity[node]))

        # >=0
        for i in range(7):
            for node in range(27):
                s.add(names['x' + str(i)][node] >= 0)

        # per app spread
        for i in range(7):
            for node in range(27):
                s.add(names['x' + str(i)][node] <= int(limit[node, i]))

        # App1 and App2 not exist
        for node in range(27):
            s.add(names['x' + str(1)][node] + names['x' + str(2)][node] <= 1)

        def handle_constraint(NUM_NODES, appid, source_batch):

            observation_original = observation.copy()

            mapping_index = []
            list_check = []

            t2 = time.time()
            for place in range(27):
                s.push()
                s.add(names['x' + str(appid)][place]   >= env.state[place][appid] + 1)

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
            for node in range(NUM_NODES):
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
        for inter_episode_index in range(NUM_CONTAINERS):

            source_batch[index_data[inter_episode_index]] -= 1

            appid = index_data[inter_episode_index]
            observation, mapping_index = handle_constraint(NUM_NODES, appid, source_batch_cpoy)
            observation[:, index_data[inter_episode_index]] += 1
            assert len(mapping_index) > 0

            observation_first_layer = np.empty([0, env.NUM_APPS], int)
            number_of_first_layer_nodes = int(NUM_NODES / nodes_per_group)  # 9
            for i in range(nodes_per_group):
                observation_new = np.sum(observation[i * number_of_first_layer_nodes:(i + 1) * number_of_first_layer_nodes], 0).reshape(1, -1)
                observation_first_layer = np.append(observation_first_layer, observation_new, 0)
            observation_first_layer[:, index_data[inter_episode_index]] += 1
            observation_first_layer = np.array(observation_first_layer).reshape(1, -1)
            observation_first_layer = np.append(observation_first_layer, index_data[inter_episode_index]).reshape(1, -1)
            observation_first_layer = np.append(observation_first_layer, np.array(source_batch)).reshape(1, -1)  # (1,29)

            action_1, prob_weights = RL_1.choose_action(observation_first_layer.copy())

            observation_copy = observation.copy()
            observation_copy = observation_copy[action_1 * number_of_first_layer_nodes: (action_1 + 1) * number_of_first_layer_nodes]
            number_of_second_layer_nodes = int(number_of_first_layer_nodes / nodes_per_group)  # 9/3 = 3
            observation_second_layer = np.empty([0, env.NUM_APPS], int)
            for i in range(nodes_per_group):
                observation_new = np.sum(observation_copy[i * number_of_second_layer_nodes:(i + 1) * number_of_second_layer_nodes], 0).reshape(1, -1)
                observation_second_layer = np.append(observation_second_layer, observation_new, 0)
            observation_second_layer[:, index_data[inter_episode_index]] += 1
            observation_second_layer = np.array(observation_second_layer).reshape(1, -1)
            observation_second_layer = np.append(observation_second_layer, index_data[inter_episode_index]).reshape(1, -1)
            observation_second_layer = np.append(observation_second_layer, np.array(source_batch)).reshape(1, -1)
            action_2, prob_weights = RL_2.choose_action(observation_second_layer.copy())

            observation_copy = observation_copy[action_2 * number_of_second_layer_nodes: (action_2 + 1) * number_of_second_layer_nodes]
            number_of_third_layer_nodes = int(number_of_second_layer_nodes / nodes_per_group)  # 3/3 = 1
            observation_third_layer = np.empty([0, env.NUM_APPS], int)
            for i in range(nodes_per_group):
                observation_new = np.sum(observation_copy[i * number_of_third_layer_nodes:(i + 1) * number_of_third_layer_nodes], 0).reshape(1, -1)
                observation_third_layer = np.append(observation_third_layer, observation_new, 0)
            observation_third_layer[:, index_data[inter_episode_index]] += 1
            observation_third_layer = np.array(observation_third_layer).reshape(1, -1)
            observation_third_layer = np.append(observation_third_layer, index_data[inter_episode_index]).reshape(1, -1)
            observation_third_layer = np.append(observation_third_layer, np.array(source_batch)).reshape(1, -1)

            action_3, prob_weights = RL_3.choose_action(observation_third_layer.copy())


            final_decision = action_1 * number_of_first_layer_nodes + action_2 * number_of_second_layer_nodes + action_3 * number_of_third_layer_nodes

            appid = index_data[inter_episode_index]
            # observation_ = env.step(action*nodes_per_group + Node_index[action], appid)
            observation_ = env.step(mapping_index[final_decision], appid)
            decision = mapping_index[final_decision]
            s.add(names['x' + str(appid)][decision] >= int(env.state[decision][appid]))
            # for i in range(number_of_node_groups):
            store_episode_1(observation_first_layer, action_1)
            store_episode_2(observation_second_layer, action_2)
            store_episode_3(observation_third_layer, action_3)
            observation = observation_.copy()  # (9,9)

        """
        After an entire allocation, calculate total throughput, reward
        """
        # start_ = time.time()
        tput_state = env.get_tput_total_env()
        tput = (sim.predict(tput_state.reshape(-1, env.NUM_APPS)) * tput_state).sum() / NUM_CONTAINERS

        # print(time.time() - start_)
        # tput = 1.0 * tput / NUM_CONTAINERS
        RL_1.store_tput_per_episode(tput, epoch_i)
        assert (np.sum(env.state, axis=1) <= params['container_limitation per node']).all()
        assert sum(sum(env.state)) == NUM_CONTAINERS
        list_check = 0
        for node in range(NUM_NODES):
            for app in range(env.NUM_APPS):
                if env.state[node, :].sum() > params['container_limitation per node'] or env.state[node, app] > 1 or (app == 1 and env.state[node, 2] > 0) or (app == 2 and env.state[node, 1] > 0):
                    list_check += env.state[node, app]
        assert (list_check == 0)

        reward_ratio = (tput)

        reward_episode_1 = [reward_ratio] * len(observation_episode_1)
        reward_episode_2 = [reward_ratio] * len(observation_episode_2)
        reward_episode_3 = [reward_ratio] * len(observation_episode_3)

        RL_1.store_training_samples_per_episode(observation_episode_1, action_episode_1, reward_episode_1, 0)
        RL_2.store_training_samples_per_episode(observation_episode_2, action_episode_2, reward_episode_2, 0)
        RL_3.store_training_samples_per_episode(observation_episode_3, action_episode_3, reward_episode_3, 0)


        """
        check_tput_quality(tput)
        """
        if names['highest_tput_' + str(tput_origimal_class)] < tput:
            highest_tput_original = names['highest_tput_' + str(tput_origimal_class)]
            optimal_range_original = names['optimal_range_' + str(tput_origimal_class)]
            names['highest_tput_' + str(tput_origimal_class)] = tput
            names['number_optimal_' + str(tput_origimal_class)] = []



            names['observation_optimal_1_' + str(tput_origimal_class)], names['action_optimal_1_' + str(tput_origimal_class)], names['reward_optimal_1_' + str(tput_origimal_class)] = [], [], []
            names['observation_optimal_2_' + str(tput_origimal_class)], names['action_optimal_2_' + str(tput_origimal_class)], names['reward_optimal_2_' + str(tput_origimal_class)] = [], [], []
            names['observation_optimal_3_' + str(tput_origimal_class)], names['action_optimal_3_' + str(tput_origimal_class)], names['reward_optimal_3_' + str(tput_origimal_class)] = [], [], []
            if UseExperienceReplay:
                names['observation_optimal_1_' + str(tput_origimal_class)].extend(observation_episode_1)
                names['action_optimal_1_' + str(tput_origimal_class)].extend(action_episode_1)
                names['reward_optimal_1_' + str(tput_origimal_class)].extend(reward_episode_1)

                names['observation_optimal_2_' + str(tput_origimal_class)].extend(observation_episode_2)
                names['action_optimal_2_' + str(tput_origimal_class)].extend(action_episode_2)
                names['reward_optimal_2_' + str(tput_origimal_class)].extend(reward_episode_2)

                names['observation_optimal_3_' + str(tput_origimal_class)].extend(observation_episode_3)
                names['action_optimal_3_' + str(tput_origimal_class)].extend(action_episode_3)
                names['reward_optimal_3_' + str(tput_origimal_class)].extend(reward_episode_3)

            names['number_optimal_' + str(tput_origimal_class)].append(NUM_CONTAINERS)
            names['optimal_range_' + str(tput_origimal_class)] = min(1.2, tput / (highest_tput_original / optimal_range_original))
        elif names['highest_tput_' + str(tput_origimal_class)] < tput * names['optimal_range_' + str(tput_origimal_class)]:

            if UseExperienceReplay:

                names['observation_optimal_1_' + str(tput_origimal_class)].extend(observation_episode_1)
                names['action_optimal_1_' + str(tput_origimal_class)].extend(action_episode_1)
                names['reward_optimal_1_' + str(tput_origimal_class)].extend(reward_episode_1)

                names['observation_optimal_2_' + str(tput_origimal_class)].extend(observation_episode_2)
                names['action_optimal_2_' + str(tput_origimal_class)].extend(action_episode_2)
                names['reward_optimal_2_' + str(tput_origimal_class)].extend(reward_episode_2)

                names['observation_optimal_3_' + str(tput_origimal_class)].extend(observation_episode_3)
                names['action_optimal_3_' + str(tput_origimal_class)].extend(action_episode_3)
                names['reward_optimal_3_' + str(tput_origimal_class)].extend(reward_episode_3)

            names['number_optimal_' + str(tput_origimal_class)].append(NUM_CONTAINERS)

        observation_episode_1, action_episode_1, reward_episode_1 = [], [], []
        observation_episode_2, action_episode_2, reward_episode_2 = [], [], []
        observation_episode_3, action_episode_3, reward_episode_3 = [], [], []


        """
        Each batch, RL.learn()
        """
        # records_per_episode = NUM_CONTAINERS * training_times_per_episode
        if (epoch_i % batch_size == 0) & (epoch_i > 1):
            if UseExperienceReplay:
                for replay_class in range(0,1):

                    reward_optimal_1 = names['reward_optimal_1_' + str(replay_class)]
                    observation_optimal_1 = names['observation_optimal_1_' + str(replay_class)]
                    action_optimal_1 = names['action_optimal_1_' + str(replay_class)]

                    reward_optimal_2 = names['reward_optimal_2_' + str(replay_class)]
                    observation_optimal_2 = names['observation_optimal_2_' + str(replay_class)]
                    action_optimal_2 = names['action_optimal_2_' + str(replay_class)]

                    reward_optimal_3 = names['reward_optimal_3_' + str(replay_class)]
                    observation_optimal_3 = names['observation_optimal_3_' + str(replay_class)]
                    action_optimal_3 = names['action_optimal_3_' + str(replay_class)]

                    number_optimal = names['number_optimal_' + str(replay_class)]

                    buffer_size = int(len(number_optimal))
                    assert sum(number_optimal) * training_times_per_episode == len(action_optimal_1)

                    if buffer_size < replay_size:
                        # TODO: if layers changes, training_times_per_episode should be modified
                        RL_1.ep_obs.extend(observation_optimal_1)
                        RL_1.ep_as.extend(action_optimal_1)
                        RL_1.ep_rs.extend(reward_optimal_1)

                        RL_2.ep_obs.extend(observation_optimal_2)
                        RL_2.ep_as.extend(action_optimal_2)
                        RL_2.ep_rs.extend(reward_optimal_2)

                        RL_3.ep_obs.extend(observation_optimal_3)
                        RL_3.ep_as.extend(action_optimal_3)
                        RL_3.ep_rs.extend(reward_optimal_3)

                    else:
                        replay_index = np.random.choice(range(buffer_size), size=replay_size, replace=False)
                        for replay_id in range(replay_size):
                            replace_start = replay_index[replay_id]
                            start_location = sum(number_optimal[:replace_start]) * training_times_per_episode
                            stop_location = sum(number_optimal[:replace_start+1]) * training_times_per_episode

                            RL_1.ep_obs.extend(observation_optimal_1[start_location: stop_location])
                            RL_1.ep_as.extend(action_optimal_1[start_location: stop_location])
                            RL_1.ep_rs.extend(reward_optimal_1[start_location: stop_location])

                            RL_2.ep_obs.extend(observation_optimal_2[start_location: stop_location])
                            RL_2.ep_as.extend(action_optimal_2[start_location: stop_location])
                            RL_2.ep_rs.extend(reward_optimal_2[start_location: stop_location])

                            RL_3.ep_obs.extend(observation_optimal_3[start_location: stop_location])
                            RL_3.ep_as.extend(action_optimal_3[start_location: stop_location])
                            RL_3.ep_rs.extend(reward_optimal_3[start_location: stop_location])

            # entropy_weight=0.1
            RL_1.learn(epoch_i, entropy_weight, True)
            RL_2.learn(epoch_i, entropy_weight, False)
            RL_3.learn(epoch_i, entropy_weight, False)

        """
        checkpoint, per 1000 episodes
        """
        if (epoch_i % 500 == 0) & (epoch_i > 1):
            highest_value = 0
            for class_replay in range(0,1):
                highest_value = names['highest_tput_' + str(class_replay)]
                optimal_number = len(names['number_optimal_' + str(class_replay)])
                print("\n epoch: %d, highest tput: %f, optimal_number: %d" % (epoch_i, highest_value,optimal_number))

            RL_1.save_session(ckpt_path_1)
            RL_2.save_session(ckpt_path_2)
            RL_3.save_session(ckpt_path_3)

            np.savez(np_path, tputs=np.array(RL_1.tput_persisit), candidate=np.array(RL_1.episode))

            """
            optimal range adaptively change
            """
            print(prob_weights)
            print(prob_weights)
            entropy_weight *= 0.5
            entropy_weight = max(entropy_weight, 0.002)
            print("time by now: ", time.time() - start_time)

        epoch_i += 1


def batch_data(NUM_CONTAINERS, NUM_APPS):
    npzfile = np.load("./data/batch_set_cpo_27node_" + str(100) + '.npz')
    batch_set = npzfile['batch_set']
    rnd_array = batch_set[hyper_parameter['batch_C_numbers'], :]
    # rnd_array = np.array([8, 9, 10, 16, 23, 17, 17])
    index_data = []

    for i in range(7):
        index_data.extend([i] * rnd_array[i])

    return rnd_array, index_data


def plot_meta(name,label_name):

    # np_path = "./checkpoint/" + name + "/optimal_file_name.npz"
    np_path = name + "/optimal_file_name.npz"

    npzfile = np.load(np_path)
    tput = npzfile['tputs']
    epoch = npzfile['candidate']
    window_size = 100
    tput_smooth = np.convolve(tput, np.ones(window_size, dtype=int), 'valid')
    epoch_smooth = np.convolve(epoch, np.ones(window_size, dtype=int), 'valid')
    plt.plot(1.0 * epoch_smooth / window_size, 1.0 * tput_smooth / window_size, '.', label=label_name)


def draw_graph_single(params):

    # plot_meta(params["path"], "RL")
    plot_meta("../results/405_testbed_multilevel_models/archive/Dynamic_large_100_128_separate_limit10_10", "RL")

    plt.legend(loc = 4)
    plt.xlabel("episode")
    plt.ylabel("throughput")
    plt.title("Affinity: app[0,1],app[7,8], Anti_Affinity: No_c < 15")
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
    parser.add_argument('--batch_choice', type=int)
    args = parser.parse_args()
    hyper_parameter['batch_C_numbers'] = args.batch_choice
    params['path'] = "lpo_27_" + str(hyper_parameter['batch_C_numbers'])

    make_path(params['path'])
    train(params)
    draw_graph_single(params)


if __name__ == "__main__":
    main()
