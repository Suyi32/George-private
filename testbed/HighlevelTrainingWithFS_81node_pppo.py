import numpy as np
import time
import os
import sys
sys.path.append("/Users/ourokutaira/Desktop/George")
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from testbed.highlevel_env_pppo import LraClusterEnv
from testbed.PolicyGradient_PPPO import PolicyGradient
import argparse
import matplotlib.pyplot as plt

"""
'--batch_choice': 0, 1, 2, ``` 30
'--container_N': 150, 200, 250
python3 HighlevelTrainingWithFS_81node_pppo.py --container_N 150 --batch_choice 0
"""

hyper_parameter = {
        'batch_C_numbers': None,
        'container_N': None
}

params = {
        'batch_size': 20,
        'epochs': 15000,
        'path': "pppo_27_fromscratch_" + str(hyper_parameter['container_N']) + "_" + str(hyper_parameter['batch_C_numbers']),
        'path_recover': "cpo_clustering_81nodes_single",
        'recover': False,
        'learning rate': 0.001,
        'nodes per group': 3,
        'number of nodes in the cluster': 3,  # 81
        'replay size': 10,
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
    # node_limit_sum = 110
    node_limit_coex = 25
    NUM_APPS = 7

    batch_size = params['batch_size']
    ckpt_path_1 = "./checkpoint/" + params['path'] + "1/model.ckpt"
    make_path(params['path'] + "1")
    env = LraClusterEnv(num_nodes=NUM_NODES)
    np_path = "./checkpoint/" + params['path'] + "/optimal_file_name.npz"
    Recover = False
    nodes_per_group = int(params['nodes per group'])
    replay_size = params['replay size']
    training_times_per_episode = 1
    safety_requirement = 0.02 * hyper_parameter['container_N']
    ifUseExternal = True

    """
    Build Network
    """
    n_actions = nodes_per_group  #: 3 nodes per group
    n_features = int(n_actions * (env.NUM_APPS + 1 + env.NUM_APPS)+ 1 + env.NUM_APPS)  #: 3*9+1 = 28
    RL_1 = PolicyGradient(
        n_actions=n_actions,
        n_features=n_features,
        learning_rate=params['learning rate'],
        suffix='1b',
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

    epoch_i = 0

    thre_entropy = 0.001
    # TODO: delete this range

    names = locals()
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

    source_batch_a, index_data_a = batch_data()  # index_data = [0,1,2,0,1,2]
    while epoch_i < params['epochs']:

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
        source_batch_first = source_batch_.copy()
        observation_first_layer = np.zeros([nodes_per_group, env.NUM_APPS], int)
        for inter_episode_index in range(NUM_CONTAINERS):

            appid = index_data[inter_episode_index]
            source_batch_first[appid] -= 1
            observation_first_layer_copy = observation_first_layer.copy()
            observation_first_layer_copy[:, appid] += 1

            observation_first_layer_copy = np.append(observation_first_layer_copy, observation_first_layer_copy > node_limit_coex, axis=1)
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

            observation_first_layer[action_1, appid] += 1

            store_episode_1(observation_first_layer_copy, action_1)

        """
        After an entire allocation, calculate total throughput, reward
        """
        env.state = observation_first_layer.copy()
        assert sum(sum(env.state)) == NUM_CONTAINERS
        assert (env.state.sum(0) == source_batch_).all()

        total_tput, list_check_sum, list_check_coex, list_check_per_app, list_check = env.get_tput_total_env()

        tput = total_tput/NUM_CONTAINERS
        reward_ratio = tput
        list_check_ratio = list_check
        safety_episode_1 = [list_check_ratio * 1.0] * len(observation_episode_1)
        reward_episode_1 = [reward_ratio * 1.0] * len(observation_episode_1)
        RL_1.store_tput_per_episode(tput, epoch_i, list_check, list_check_coex, list_check_sum, list_check_per_app)
        RL_1.store_training_samples_per_episode(observation_episode_1, action_episode_1, reward_episode_1, safety_episode_1)
        """
        check_tput_quality(tput)
        """
        if list_check <= safety_requirement*0.5:
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
                names['number_optimal_' + str(tput_origimal_class)].append(NUM_CONTAINERS)
                names['safety_optimal_1_' + str(tput_origimal_class)].extend(safety_episode_1)
                names['reward_optimal_1_' + str(tput_origimal_class)].extend(reward_episode_1)
                names['optimal_range_' + str(tput_origimal_class)] = 1.05

            elif names['highest_tput_' + str(tput_origimal_class)] < tput * names['optimal_range_' + str(tput_origimal_class)]:
                names['observation_optimal_1_' + str(tput_origimal_class)].extend(observation_episode_1)
                names['action_optimal_1_' + str(tput_origimal_class)].extend(action_episode_1)
                names['number_optimal_' + str(tput_origimal_class)].append(NUM_CONTAINERS)
                names['safety_optimal_1_' + str(tput_origimal_class)].extend(safety_episode_1)
                names['reward_optimal_1_' + str(tput_origimal_class)].extend(reward_episode_1)

        observation_episode_1, action_episode_1, reward_episode_1, safety_episode_1 = [], [], [], []

        """
        Each batch, RL.learn()
        """
        if (epoch_i % batch_size == 0) & (epoch_i > 1):
            for replay_class in range(0,10):

                number_optimal = names['number_optimal_' + str(replay_class)]
                reward_optimal_1 = names['reward_optimal_1_' + str(replay_class)]
                safety_optimal_1 = names['safety_optimal_1_' + str(replay_class)]
                observation_optimal_1 = names['observation_optimal_1_' + str(replay_class)]
                action_optimal_1 = names['action_optimal_1_' + str(replay_class)]
                buffer_size = int(len(number_optimal))

                if buffer_size < replay_size:
                    # TODO: if layers changes, training_times_per_episode should be modified
                    RL_1.ep_obs.extend(observation_optimal_1)
                    RL_1.ep_as.extend(action_optimal_1)
                    RL_1.ep_rs.extend(reward_optimal_1)
                    RL_1.ep_ss.extend(safety_optimal_1)

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

            #
            optim_case = RL_1.learn(epoch_i, thre_entropy, Ifprint=True)


        """
        checkpoint, per 1000 episodes
        """
        if (epoch_i % 2000 == 0) & (epoch_i > 1):
            for class_replay in range(0,10):
                highest_value = names['highest_tput_' + str(class_replay)]
                print("\n epoch: %d, highest tput: %f" % (epoch_i, highest_value))

            RL_1.save_session(ckpt_path_1)
            np.savez(np_path, tputs=np.array(RL_1.tput_persisit), candidate=np.array(RL_1.episode), vi_coex=np.array(RL_1.coex_persisit), vio_sum=np.array(RL_1.sum_persisit), vio_persis=np.array(RL_1.safe_persisit))
            """
            optimal range adaptively change
            """
            for class_replay in range(1, 10):
                number_optimal = names['number_optimal_' + str(class_replay)]
                count_size = int(len(number_optimal))

                if (count_size > 100):
                    names['optimal_range_' + str(class_replay)] *= 0.99
                    names['optimal_range_' + str(class_replay)] = max(names['optimal_range_' + str(class_replay)], 1.01)
                    start_location = sum(names['number_optimal_' + str(class_replay)][:-10]) * training_times_per_episode
                    names['observation_optimal_1_' + str(class_replay)] = names['observation_optimal_1_' + str(class_replay)][start_location:]
                    names['action_optimal_1_' + str(class_replay)] = names['action_optimal_1_' + str(class_replay)][start_location:]
                    names['number_optimal_' + str(class_replay)] = names['number_optimal_' + str(class_replay)][-10:]
                    names['safety_optimal_1_' + str(class_replay)] = names['safety_optimal_1_' + str(class_replay)][start_location:]
                    names['reward_optimal_1_' + str(class_replay)] = names['reward_optimal_1_' + str(class_replay)][start_location:]

                print("optimal_range:", names['optimal_range_' + str(class_replay)])

            print(prob_weights)
            if optim_case > 0:
                thre_entropy *= 0.5
            thre_entropy = max(thre_entropy, 0.001)

        epoch_i += 1
        if epoch_i>5:
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
    parser.add_argument('--container_N', type=int)
    parser.add_argument('--batch_choice', type=int)
    args = parser.parse_args()
    hyper_parameter['batch_C_numbers'] = args.batch_choice
    hyper_parameter['container_N'] = args.container_N
    params['path'] = "pppo_27_fromscratch_" + str(hyper_parameter['container_N']) + "_" + str(hyper_parameter['batch_C_numbers'])
    make_path(params['path'])
    train(params)
    draw_graph_single(params)


if __name__ == "__main__":
    main()
