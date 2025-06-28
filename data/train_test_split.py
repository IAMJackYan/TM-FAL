from glob import glob
import pdb
import json
import random
import os

def split_prostate():
    if not os.path.exists('data_split/FedProstate/'):
        os.makedirs('data_split/FedProstate/')
        
    for i in range(6):
        data_list = glob('/home/ylyan/data/FedProstate_npy/client{}/*'.format(i))
        random.shuffle(data_list)   

        data_len = len(data_list)   # 
        test_len = int(0.2*data_len)

        test_list = data_list[:test_len]
        train_list = data_list[test_len:]

        train_slice_list = []
        for train_case in train_list:
            train_slice_list.extend(glob('{}/*'.format(train_case)))
        
        with open("data_split/FedProstate/client{}_train.txt".format(i+1), "w") as f: 
            json.dump(train_slice_list, f)

        test_slice_list = []
        for test_case in test_list:
            test_slice_list.extend(glob('{}/*'.format(test_case)))
        with open("data_split/FedProstate/client{}_test.txt".format(i+1), "w") as f: 
            json.dump(test_slice_list, f)


def split_dataset(dataset, client_num):
    if not os.path.exists('data_split/{}/'.format(dataset)):
        os.makedirs('data_split/{}/'.format(dataset))

    for i in range(1, client_num+1):   
        data_list = glob('/home/ylyan/data/{}/client{}/*'.format(dataset, i))

        data_len = len(data_list)
        test_len = int(0.2*data_len)

        test_list = data_list[:test_len]
        train_list = data_list[test_len:]

        with open("data_split/{}/client{}_train.txt".format(dataset, i), "w") as f: 
            json.dump(train_list, f)
        with open("data_split/{}/client{}_test.txt".format(dataset, i), "w") as f: 
            json.dump(test_list, f)

if __name__ == '__main__':
    client_name = ['client1', 'client2', 'client3', 'client4', 'client5', 'client6']
    object_lens = [225, 306, 134, 387, 337, 152]
    # split_dataset('FedPolyp_npy', 4)
    from tqdm import tqdm
    for i in tqdm(range(100000)):
        split_prostate()
        lens = []
        for i in range(6):
            with open("data_split/FedProstate/{}_{}.txt".format(client_name[i], 'train'), "r") as f: 
                data_list1 = json.load(f)
                lens.append(len(data_list1))
        
        if lens==object_lens:
            break

    # split_dataset('FedFundus_npy', 4)