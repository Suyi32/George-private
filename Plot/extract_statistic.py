import numpy as np

def extract(name):
    np_path = "./second_obj/" + name + "/optimal_file_name.npz"
    npzfile = np.load(np_path)
    tput = npzfile['tputs'][0:50000]
    vio = npzfile['vi_sum'][0:50000]
    epoch = npzfile['candidate']
    epoch = epoch[0:50000]
    epoch = epoch
    # We define the "optimal point" during the training as follows

    # If the min_vale of violation <= 5%, we find the maximum tput under the constraint that vio<5%
    if min(vio) <= 5:
        max_tput = max(tput[np.where(vio<=5)[0]])
        index = np.where((max_tput == tput) & (vio<=5))[0][0]
        epispode_ = epoch[index]
        vio_ = vio[index]

    # Otherwise, there is no data that satisfies vio<=5%, then we find the minimum vio
    else:
        vio_ = min(vio)
        index = np.where(vio_ == vio)[0][0]
        max_tput = tput[index]
        epispode_ = epoch[index]

    # we assume the training is terminated 50 episodes after the "optimal point" is obtained
    epispode_ += 50
    return max_tput, vio_, epispode_


def save_data():
    max_tput_set, vio_set, epispode_set = [], [], []
    for i in range(30):#
        max_tput, vio_, epispode_ = extract("pppo_27_" +str(i))
        max_tput_set.append(max_tput)
        vio_set.append(vio_)
        epispode_set.append(epispode_)

    import pandas as pd
    save = pd.DataFrame(max_tput_set)
    save.to_csv(
        './' + "secondobj_pppo_" + 'tput.csv',
        index=False, header=False)

    save = pd.DataFrame(vio_set)
    save.to_csv(
        './' + "secondobj_pppo_" + 'vio.csv',
        index=False,
        header=False)

    save = pd.DataFrame(epispode_set)
    save.to_csv(
        './' + "secondobj_pppo_" + 'time.csv',
        index=False,
        header=False)