import os
import re
from copy import copy
from typing import List, Tuple
import pandas as pd
from torcheeg.datasets.module.base_dataset import BaseDataset


class LeaveOneSubjectOut:
    def __init__(self, split_path: str = './split', split_rest_path: str = None):
        self.split_path = split_path
        self.split_rest_path = split_rest_path

    def split_info_constructor(self, info: pd.DataFrame, path) -> None:
        subjects = list(set(info['subject_id']))

        for test_subject in subjects:
            train_subjects = subjects.copy()
            train_subjects.remove(test_subject)

            train_info = []
            for train_subject in train_subjects:
                train_info.append(info[info['subject_id'] == train_subject])

            train_info = pd.concat(train_info)
            test_info = info[info['subject_id'] == test_subject]

            train_info.to_csv(os.path.join(path, f'train_subject_{test_subject}.csv'), index=False)
            test_info.to_csv(os.path.join(path, f'test_subject_{test_subject}.csv'), index=False)

    @property
    def subjects(self) -> List:
        indice_files = list(os.listdir(self.split_path))

        def indice_file_to_subject(indice_file):
            return re.findall(r'subject_(.*).csv', indice_file)[0]

        subjects = list(set(map(indice_file_to_subject, indice_files)))
        subjects.sort()
        return subjects

    def split(self, dataset: BaseDataset, *args) -> Tuple[BaseDataset, BaseDataset]:
        if not os.path.exists(self.split_path):
            os.makedirs(self.split_path)
            self.split_info_constructor(dataset.info, self.split_path)
        if self.split_rest_path:
            dataset_rest = args[0]
            if not os.path.exists(self.split_rest_path):
                os.makedirs(self.split_rest_path)
                self.split_info_constructor(dataset_rest.info, self.split_rest_path)

        subjects = self.subjects
        for subject in subjects:
            train_info = pd.read_csv(os.path.join(self.split_path, f'train_subject_{subject}.csv'))
            test_info = pd.read_csv(os.path.join(self.split_path, f'test_subject_{subject}.csv'))
            train_dataset = copy(dataset)
            train_dataset.info = train_info
            test_dataset = copy(dataset)
            test_dataset.info = test_info

            if self.split_rest_path:
                train_rest_info = pd.read_csv(os.path.join(self.split_rest_path, f'train_subject_{subject}.csv'))
                test_rest_info = pd.read_csv(os.path.join(self.split_rest_path, f'test_subject_{subject}.csv'))
                train_rest_dataset = copy(dataset_rest)
                train_rest_dataset.info = train_rest_info
                test_rest_dataset = copy(dataset_rest)
                test_rest_dataset.info = test_rest_info

            if not self.split_rest_path:
                yield train_dataset, test_dataset
            else:
                yield train_dataset, test_dataset, train_rest_dataset, test_rest_dataset
