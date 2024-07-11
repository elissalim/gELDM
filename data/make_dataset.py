import numpy as np
import torch
import torcheeg.datasets.moabb as moabb_dataset
from torcheeg import transforms
from moabb.paradigms import MotorImagery
from torch.utils.data import Dataset, DataLoader

from data.bnci import BNCI2014001, BNCI2014004
from data.leave_one_subject_out import LeaveOneSubjectOut

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_item = self.data[idx]
        label_item = self.labels[idx].type(torch.long)
        return data_item, label_item

def load_data(dataset, classes, signal_type):
    if dataset == 'BNCI2014001':
        dataset = BNCI2014001(signal_type)
        dataset.subject_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        dataset.event_id = classes
        if signal_type == "MI":
            data_path = './data/moabb/bnci_2014-001/'
            t_map = {'left_hand': 0, 'right_hand': 1, "feet": 2, "tongue": 3}
        elif signal_type == "REST":
            data_path = './data/moabb/bnci_2014-001_rest/'
            t_map = {'left_hand': 4, 'right_hand': 4, "feet": 4, "tongue": 4}
    elif dataset == 'BNCI2014004':
        dataset = BNCI2014004(signal_type)
        dataset.subject_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        dataset.event_id = classes
        if signal_type == "MI":
            data_path = './data/moabb/bnci_2014-004/'
            t_map = {'left_hand': 0, 'right_hand': 1}
        elif signal_type == "REST":
            data_path = './data/moabb/bnci_2014-004_rest/'
            t_map = {'left_hand': 2, 'right_hand': 2}
    else:
        raise ValueError("Dataset not found!")

    paradigm = MotorImagery()
    dataset = moabb_dataset.MOABBDataset(
        dataset=dataset,
        paradigm=paradigm,
        io_path=data_path,
        online_transform=transforms.ToTensor(),
        label_transform=transforms.Compose([transforms.Select('label'), transforms.Mapping(t_map)])
    )
    return dataset

def train_test_split(x_y_all, train_idxs, test_idxs, device):
    if type(x_y_all) is tuple:
        data_0, label_0 = x_y_all[0], x_y_all[1]
    else:
        data_0, label_0 = x_y_all, x_y_all

    for sample_id in range(len(train_idxs)):
        if type(x_y_all) is tuple:
            data, label = data_0[train_idxs[sample_id]], label_0[train_idxs[sample_id]]
        else:
            data, label = data_0[train_idxs[sample_id]][0], label_0[train_idxs[sample_id]][1]

        if sample_id == 0:
            train_data = data.unsqueeze(0)
            train_labels = [label]
        else:
            train_data = torch.cat((train_data, data.unsqueeze(0)), 0)
            train_labels.append(label)

    for sample_id in range(len(test_idxs)):
        if type(x_y_all) is tuple:
            data, label = data_0[test_idxs[sample_id]], label_0[test_idxs[sample_id]]
        else:
            data, label = data_0[test_idxs[sample_id]][0], label_0[test_idxs[sample_id]][1]

        if sample_id == 0:
            test_data = data.unsqueeze(0)
            test_labels = [label]
        else:
            test_data = torch.cat((test_data, data.unsqueeze(0)), 0)
            test_labels.append(label)
    return train_data, torch.tensor(train_labels, dtype=torch.int).to(device), test_data, torch.tensor(test_labels, dtype=torch.int).to(device)

def process_data(dataset_name, subject, device):
    train_batch_size, test_batch_size = 16, 16
    if dataset_name == "BNCI2014001":
        classes = {"left_hand": 1, "right_hand": 2, "feet": 3, "tongue": 4}
        cv = LeaveOneSubjectOut(split_path="./data/split_2a", split_rest_path="./data/split_rest_2a")
    elif dataset_name == "BNCI2014004":
        classes = {"left_hand": 1, "right_hand": 2}
        cv = LeaveOneSubjectOut(split_path="./data/split_2b", split_rest_path="./data/split_rest_2b")

    mi_ds = load_data(dataset_name, classes, "MI")
    rest_ds = load_data(dataset_name, classes, "REST")
    num_classes = len(classes)
    for idx, (_, sub_dataset, _, sub_dataset_rest) in enumerate(cv.split(mi_ds, rest_ds)):
        if idx == subject - 1:
            if dataset_name == "BNCI2014001":
                new_tensor = torch.zeros_like(sub_dataset_rest[0][0])
                new_tensor[:, :500] = sub_dataset_rest[0][0][:, :500]
                all_x = torch.cat((new_tensor.unsqueeze(0).unsqueeze(0), sub_dataset[0][0].unsqueeze(0).unsqueeze(0)), 1)
            elif dataset_name == "BNCI2014004":
                all_x = torch.cat((sub_dataset_rest[0][0].unsqueeze(0).unsqueeze(0), sub_dataset[0][0].unsqueeze(0).unsqueeze(0)), 1)
            all_y = [sub_dataset[0][1]]

            for i in range(1, len(sub_dataset)):
                if dataset_name == "BNCI2014001":
                    new_tensor = torch.zeros_like(sub_dataset_rest[i][0])
                    new_tensor[:, :500] = sub_dataset_rest[i][0][:, :500]
                    all_x = torch.cat((all_x, torch.cat((new_tensor.unsqueeze(0).unsqueeze(0), sub_dataset[i][0].unsqueeze(0).unsqueeze(0)), 1)), 0)
                elif dataset_name == "BNCI2014004":
                    all_x = torch.cat((all_x, torch.cat((sub_dataset_rest[i][0].unsqueeze(0).unsqueeze(0), sub_dataset[i][0].unsqueeze(0).unsqueeze(0)), 1)), 0)
                all_y.append(sub_dataset[i][1])

            if num_classes == 4:
                indices = [id for id, x in enumerate(all_y) if x == 0 or x == 1 or x == 2 or x == 3]
            elif num_classes == 2:
                indices = [id for id, x in enumerate(all_y) if x == 0 or x == 1]
            all_x = all_x[indices]
            filtered_y = [all_y[index] for index in indices]
            all_y = torch.tensor(filtered_y, dtype=torch.int).to(device)
            all_y = torch.tensor(all_y, dtype=torch.int).to(device)
            all_x = all_x.to(device)

            train_ids = np.random.choice(list(range(len(all_x))), size=int(len(all_x) * 0.8), replace=False)
            test_ids = np.array(list(set(list(range(len(all_x)))).difference(train_ids)))
            train_x, train_y, test_x, test_y = train_test_split((all_x, all_y), train_ids, test_ids, device)
            train_ds = CustomDataset(train_x, train_y)
            test_ds = CustomDataset(test_x, test_y)
            train_loader = DataLoader(train_ds, batch_size=train_batch_size, shuffle=True)
            test_loader = DataLoader(test_ds, batch_size=test_batch_size, shuffle=True)
            return train_loader, test_loader, num_classes
        else:
            continue
