import time
import argparse
import random
import numpy as np
import torch
import pandas as pd

from models.geldm import *

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True

########################################################################################################################

if __name__ == "__main__":
    for d_id in range(2):
        parser = argparse.ArgumentParser(description="Train a machine learning model")
        parser.add_argument("--num_subjects", type=int, default=9, help="Number of subjects to process")
        parser.add_argument("--device", type=str, default="cuda:0", help="Device to use (default: cuda:0)")

        args = parser.parse_args()
        timestamp = int(time.time())
        dataset_idx = d_id
        dataset_names = ["BNCI2014001", "BNCI2014004"]  # 2a, 2b
        channels = [22, 3]
        args.dataset_name = dataset_names[dataset_idx]
        args.channel = channels[dataset_idx]
        args.num_epochs = 500
        args.model_name = "gELDM"
        args.timestamp = timestamp
        args.early_stopping = True
        overall_result = pd.DataFrame(columns=['subject', 'epoch', 'train/test', 'accuracy', 'f1_score'])

        for i in range(1, args.num_subjects + 1):
            args.subject = i
            args.epoch_result = pd.DataFrame(columns=['subject', 'epoch', 'train_acc', 'test_acc', 'train_loss'])
            subject, epoch, accuracy_test, f1_score_test, epoch_result, train_loss = train(args)
            overall_result = pd.concat([overall_result, pd.DataFrame({'subject': subject, 'epoch': epoch, 'train/test': 'test', 'accuracy': [accuracy_test], 'f1_score': [f1_score_test], 'train_loss': [train_loss]})], axis=0, ignore_index=True)
            epoch_result.to_csv(f'./results/{args.dataset_name}_{args.model_name}_{args.timestamp}/s{subject}_epoch.csv', index=False)
        overall_result.to_csv(f'./results/{args.dataset_name}_{args.model_name}_{args.timestamp}/results.csv', index=False)


