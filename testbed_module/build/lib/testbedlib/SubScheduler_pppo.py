import numpy as np
from testbedlib.cluster_env import LraClusterEnv
from testbedlib.PolicyGradient_PCPO_PPO import PolicyGradient
from z3 import *
import z3

params = {
        # 'path': "Dynamic_large_100",
        # 'path': "Dynamic_large_100_limit10",
        # 'number of containers': 81,
        'learning rate': 0.015,
        'nodes per group': 3,
        'number of nodes in the cluster': 27,
        'container_limitation per node':8
    }

app_node_set = np.array([
     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26],
     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26],
     [2, 3, 5, 6, 7, 11, 12, 18, 20, 22, 23, 24, 25, 26],
     [0, 2, 8, 9, 19, 23, 24, 25, 26],
     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26],
     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26],
     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]])
for idx in range(len(app_node_set)):
    print("[INFO] No. of nodes for App {}: {}".format(idx, len(app_node_set[idx]) ))

def handle_constraint(observation, NUM_NODES):

    observation_original = observation.copy()
    mapping_index = []
    # TODO: we could add more constraints here
    list_check = observation[:, :].sum(1) > params['container_limitation per node'] - 1   # >8

    if sum(list_check) == NUM_NODES:
        return [],[]

    good_index = np.where(list_check == False)[0]
    length = len(good_index)
    index_replace = 0
    for node in range(NUM_NODES):
        if list_check[node]:  # bad node
            # index_this_replace = good_index[np.random.randint(length)]
            index_this_replace = good_index[index_replace % length]
            index_replace += 1
            observation[node] = observation_original[index_this_replace]
            mapping_index.append(index_this_replace)
        else:
            mapping_index.append(node)
            observation[node] = observation_original[node]

    return observation, mapping_index

class NineNodeAPI():

    def __init__(self, path_name, surffix, path_surffix):
        """
        parameters set
        """
        self.NUM_NODES = params['number of nodes in the cluster']
        # self.NUM_CONTAINERS = params['number of containers']

        # self.sim = Simulator()
        self.env = LraClusterEnv(num_nodes=self.NUM_NODES)

        ckpt_path_1 = path_surffix + path_name + "1" + "/model.ckpt"
        ckpt_path_2 = path_surffix + path_name + "2" + "/model.ckpt"
        ckpt_path_3 = path_surffix + path_name + "3" + "/model.ckpt"
        self.nodes_per_group = int(params['nodes per group'])
        # self.number_of_node_groups = int(self.NUM_NODES / self.nodes_per_group)
        """
        Build Network
        """
        self.n_actions = self.nodes_per_group  #: 3 nodes per group
        self.n_features = int(self.n_actions * (self.env.NUM_APPS + 1 + self.env.NUM_APPS) + 1 + self.env.NUM_APPS)
        #: 29

        self.RL_1 = PolicyGradient(n_actions=self.n_actions, n_features=self.n_features, learning_rate=params['learning rate'], suffix=surffix + '1a')

        self.RL_2 = PolicyGradient(n_actions=self.n_actions, n_features=self.n_features, learning_rate=params['learning rate'], suffix=surffix + '2a')

        self.RL_3 = PolicyGradient(n_actions=self.n_actions, n_features=self.n_features, learning_rate=params['learning rate'], suffix=surffix + '3a')
        self.RL_1.restore_session(ckpt_path_1)
        self.RL_2.restore_session(ckpt_path_2)
        self.RL_3.restore_session(ckpt_path_3)

        self.observation_episode_1, self.action_episode_1, self.reward_episode_1, self.safety_episode_1 = [], [], [], []
        self.observation_optimal_1, self.action_optimal_1, self.reward_optimal_1, self.safety_optimal_1 = [], [], [], []

        self.observation_episode_2, self.action_episode_2, self.reward_episode_2, self.safety_episode_2 = [], [], [], []
        self.observation_optimal_2, self.action_optimal_2, self.reward_optimal_2, self.safety_optimal_2 = [], [], [], []

        self.observation_episode_3, self.action_episode_3, self.reward_episode_3, self.safety_episode_3 = [], [], [], []
        self.observation_optimal_3, self.action_optimal_3, self.reward_optimal_3, self.safety_optimal_3 = [], [], [], []

    def batch_data(self, rnd_array):
        index_data = []
        for i in range(7):
            index_data.extend([i] * rnd_array[i])
        return rnd_array, index_data

    def batch_data_sub(self, rnd_array):

        rnd_array = rnd_array.copy()
        index_data = []
        for i in range(7):
            index_data.extend([i] * int(rnd_array[i]))

        return rnd_array, index_data

    def store_episode_1(self, observations, actions):
        self.observation_episode_1.append(observations)
        self.action_episode_1.append(actions)

    def store_episode_2(self, observations, actions):
        self.observation_episode_2.append(observations)
        self.action_episode_2.append(actions)

    def store_episode_3(self, observations, actions):
        self.observation_episode_3.append(observations)
        self.action_episode_3.append(actions)

    def get_total_tput(self, rnd_array):

        # assert sum(rnd_array) == 81
        source_batch_, index_data = self.batch_data(rnd_array.astype(int))  # index_data = [0,1,2,0,1,2]
        env = LraClusterEnv(num_nodes=self.NUM_NODES)
        ilp_dict = {}
        for i in range(7):
            ilp_dict['x' + str(i)] = z3.IntVector('x' + str(i), 3)
        observation = env.reset().copy()  # (9,9)
        source_batch = source_batch_.copy()
        nodes_per_group = int(params['nodes per group'])
        NUM_CONTAINERS = int(sum(rnd_array))

        """
        Episode
        """
        def handle_constraint(observation_now, appid_now, s):

            observation_original = observation_now.copy()

            mapping_index = []
            list_check = []

            for place in range(3):
                s.push()
                s.add(ilp_dict['x' + str(appid_now)][place] >= int(observation_now[place][appid_now]) + 1)
                if s.check() == z3.sat:
                    list_check.append(False)
                else:
                    list_check.append(True)
                s.pop()

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
        first layer
        """


        total = source_batch_.copy()
        limit = (1 * 9)
        capicity = (8 * 9)  # 3
        s_first = Solver()
        # app sum == batch
        for i in range(7):
            s_first.add(z3.Sum(ilp_dict['x' + str(i)]) == int(total[i]))
        # node capacity
        for node in range(3):
            s_first.add(z3.Sum([ilp_dict['x' + str(i)][node] for i in range(7)]) <= int(capicity))
        # >=0
        for i in range(7):
            for node in range(3):
                s_first.add(ilp_dict['x' + str(i)][node] >= 0)
        # per app spread
        for i in range(7):
            for node in range(3):
                s_first.add(ilp_dict['x' + str(i)][node] <= limit)
        # App1 and App2 not exist
        # for node in range(3):
        #     s_first.add(ilp_dict['x' + str(1)][node] + ilp_dict['x' + str(2)][node] <= limit)



        source_batch_first = source_batch_.copy()
        observation_first_layer = np.zeros([nodes_per_group, env.NUM_APPS], int)
        for inter_episode_index in range(NUM_CONTAINERS):
            appid = index_data[inter_episode_index]
            observation_first_layer_copy, mapping_index = handle_constraint(observation_first_layer, appid, s_first)
            assert len(mapping_index) > 0

            source_batch_first[appid] -= 1
            # observation_first_layer_copy = observation_first_layer.copy()
            observation_first_layer_copy[:, appid] += 1
            observation_first_layer_copy = np.append(observation_first_layer_copy, observation_first_layer_copy > 9 * 2, axis=1)
            observation_first_layer_copy = np.append(observation_first_layer_copy, observation_first_layer_copy.sum(axis=1).reshape(nodes_per_group, 1), axis=1)
            observation_first_layer_copy = np.array(observation_first_layer_copy).reshape(1, -1)
            observation_first_layer_copy = np.append(observation_first_layer_copy, appid).reshape(1, -1)
            observation_first_layer_copy = np.append(observation_first_layer_copy, np.array(source_batch_first)).reshape(1, -1)
            action_1, prob_weights = self.RL_1.choose_action(observation_first_layer_copy.copy())
            decision = mapping_index[action_1]
            observation_first_layer[decision, appid] += 1
            s_first.add(ilp_dict['x' + str(appid)][decision] >= int(observation_first_layer[decision][appid]))
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
            limit = (1 * 3)
            capicity = (8 * 3)  # 3
            s_second = Solver()
            # app sum == batch
            for i in range(7):
                s_second.add(z3.Sum(ilp_dict['x' + str(i)]) == int(total[i]))
            # node capacity
            for node in range(3):
                s_second.add(z3.Sum([ilp_dict['x' + str(i)][node] for i in range(7)]) <= int(capicity))
            # >=0
            for i in range(7):
                for node in range(3):
                    s_second.add(ilp_dict['x' + str(i)][node] >= 0)
            # per app spread
            for i in range(7):
                for node in range(3):
                    s_second.add(ilp_dict['x' + str(i)][node] <= limit)
            # App1 and App2 not exist
            # for node in range(3):
            #     s_second.add(ilp_dict['x' + str(1)][node] + ilp_dict['x' + str(2)][node] <= limit)

            source_batch_second, index_data = self.batch_data_sub(rnd_array)
            observation_second_layer = np.zeros([nodes_per_group, env.NUM_APPS], int)
            NUM_CONTAINERS_second = sum(source_batch_second)
            number_cont_second_layer.append(NUM_CONTAINERS_second)

            for inter_episode_index in range(NUM_CONTAINERS_second):

                appid = index_data[inter_episode_index]
                observation_second_layer_copy, mapping_index = handle_constraint(observation_second_layer, appid, s_second)
                assert len(mapping_index) > 0

                source_batch_second[appid] -= 1
                # observation_second_layer_copy = observation_second_layer.copy()
                observation_second_layer_copy[:, appid] += 1
                observation_second_layer_copy = np.append(observation_second_layer_copy, observation_second_layer_copy > 3 * 2, axis=1)
                observation_second_layer_copy = np.append(observation_second_layer_copy, observation_second_layer_copy.sum(axis=1).reshape(nodes_per_group, 1), axis=1)
                observation_second_layer_copy = np.array(observation_second_layer_copy).reshape(1, -1)
                observation_second_layer_copy = np.append(observation_second_layer_copy, appid).reshape(1, -1)
                observation_second_layer_copy = np.append(observation_second_layer_copy, np.array(source_batch_second)).reshape(1, -1)

                action_2, prob_weights = self.RL_2.choose_action(observation_second_layer_copy.copy())
                decision = mapping_index[action_2]
                observation_second_layer[decision, appid] += 1
                s_second.add(ilp_dict['x' + str(appid)][decision] >= int(observation_second_layer[decision][appid]))

            observation_second_layer_aggregation = np.append(observation_second_layer_aggregation, observation_second_layer, 0)
            assert (np.sum(observation_second_layer, axis=1) <= params['container_limitation per node'] * 3).all()
            assert sum(sum(observation_second_layer)) == NUM_CONTAINERS_second

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
            s_third = Solver()
            # app sum == batch
            for i in range(7):
                s_third.add(z3.Sum(ilp_dict['x' + str(i)]) == int(total[i]))
            # node capacity
            for node in range(3):
                s_third.add(z3.Sum([ilp_dict['x' + str(i)][node] for i in range(7)]) <= int(capicity))
            # >=0
            for i in range(7):
                for node in range(3):
                    s_third.add(ilp_dict['x' + str(i)][node] >= 0)
            # per app spread
            for i in range(7):
                for node in range(3):
                    s_third.add(ilp_dict['x' + str(i)][node] <= limit)
            # App1 and App2 not exist
            # for node in range(3):
            #     s_third.add(ilp_dict['x' + str(1)][node] + ilp_dict['x' + str(2)][node] <= limit)


            source_batch_third, index_data = self.batch_data_sub(rnd_array)
            observation_third_layer = np.zeros([nodes_per_group, env.NUM_APPS], int)
            NUM_CONTAINERS_third = sum(source_batch_third)
            number_cont_third_layer.append(NUM_CONTAINERS_third)

            for inter_episode_index in range(NUM_CONTAINERS_third):
                appid = index_data[inter_episode_index]
                observation_third_layer_copy, mapping_index = handle_constraint(observation_third_layer, appid, s_third)
                assert len(mapping_index) > 0

                source_batch_third[appid] -= 1
                # observation_third_layer_copy = observation_third_layer.copy()
                observation_third_layer_copy[:, appid] += 1

                observation_third_layer_copy = np.append(observation_third_layer_copy, observation_third_layer_copy > 1 * 2, axis=1)
                observation_third_layer_copy = np.append(observation_third_layer_copy, observation_third_layer_copy.sum(axis=1).reshape(nodes_per_group, 1), axis=1)
                observation_third_layer_copy = np.array(observation_third_layer_copy).reshape(1, -1)
                observation_third_layer_copy = np.append(observation_third_layer_copy, appid).reshape(1, -1)
                observation_third_layer_copy = np.append(observation_third_layer_copy, np.array(source_batch_third)).reshape(1, -1)

                action_3, prob_weights = self.RL_3.choose_action(observation_third_layer_copy.copy())
                decision = mapping_index[action_3]
                observation_third_layer[decision, appid] += 1
                s_third.add(ilp_dict['x' + str(appid)][decision] >= int(observation_third_layer[decision][appid]))

            observation_third_layer_aggregation = np.append(observation_third_layer_aggregation, observation_third_layer, 0)
            assert (np.sum(observation_third_layer, axis=1) <= params['container_limitation per node'] * 1).all()
            assert sum(sum(observation_third_layer)) == NUM_CONTAINERS_third

        env.state = observation_third_layer_aggregation.copy()
        assert sum(sum(env.state)) == NUM_CONTAINERS
        assert (env.state.sum(0) == source_batch_).all()
        """
        After an entire allocation, calculate total throughput, reward
        """
        # state = env.state
        # assert sum(sum(self.env.state)) == 81

        return env.state

