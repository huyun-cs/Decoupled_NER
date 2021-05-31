# coding: UTF-8

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import time
import torch
import numpy as np
from train_eval_3_loss import train, init_network,only_test
from importlib import import_module
import argparse
from utils_3_loss_share import build_dataset, build_iterator, get_time_dif

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=True, help='choose a model: Bert, ERNIE')
args = parser.parse_args()



if __name__ == '__main__':
    dataset = 'deNER'  

    model_name = args.model  # bert
    x = import_module('models.' + model_name)
    config = x.Config(dataset)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  


    print("train_path，\t",config.train_path)
    print("dev_path，\t",config.dev_path)
    print("test_path，\t",config.test_path)
    print("save_path，\t",config.save_path)
    print("num_classes，\t",config.num_classes)
    print("batch_size，\t",config.batch_size)
    print("learning_rate，\t",config.learning_rate)
    print("bert_path_share，\t",config.bert_path_share)

    start_time = time.time()
    print("Loading data...")
    train_data, dev_data, test_data = build_dataset(config)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    model = x.Model(config).to(config.device)
    only_test(config, model, test_iter)
