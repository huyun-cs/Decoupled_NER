# coding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from utils_3_loss_share import get_time_dif
from transformers.optimization import AdamW
from transformers import get_linear_schedule_with_warmup
import pdb
from computer_F import compute_P_R_F, no_nested_process
from Focal_Loss import focal_loss

if torch.cuda.is_available():
    APLHA = torch.Tensor([0.5]).cuda()
    BETA = torch.Tensor([0.5]).cuda()
else:
    APLHA = torch.Tensor([0.0])
    BETA = torch.Tensor([0.0])


Results_write_path = "compute_F_write.txt"
Seg_path = "./deNER/ltp.simplify.ner.dev.mask.j.txt"
Postive_path = "./deNER/dev.pos.maskpos"


####### ACL 2020 temp test
Seg_path_test = "./deNER/part_test/ltp_simplify_ner_test_mask_j_data_size_02.txt"
Postive_path_test = "./deNER/test.pos.maskpos"


def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if len(w.size()) < 2:
                continue
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass


def train(config, model, train_iter, dev_iter, test_iter):
    start_time = time.time()
    model.train()
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]



    optimizer = AdamW(model.parameters(), lr=config.learning_rate, correct_bias=False)



    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=50,
                                                num_training_steps=len(train_iter) * config.num_epochs)

    total_batch = 0  
    dev_best_acc = 0
    dev_best_F = 0
    last_improve = 0 
    flag = False 
    model.train()
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        for i, (trains_entity, trains_context, labels) in enumerate(train_iter):
            outputs, outputs_entity, outputs_context = model(trains_entity, trains_context)
            model.zero_grad()
            # loss = torch.nn.functional.cross_entropy(outputs, labels)
            fl = focal_loss(alpha= [1.0, 1.0, 1.0, 1.0, 1.0], gamma=0, num_classes = 5, size_average=True)
            loss = fl(outputs, labels)
            loss_entity = torch.nn.functional.cross_entropy(outputs_entity, labels)
            loss_context = torch.nn.functional.cross_entropy(outputs_context, labels)
            final_loss = loss + APLHA * loss_entity + BETA * loss_context
            final_loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            if total_batch % 100 == 0:
               
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                dev_acc, dev_loss, dev_P, dev_R, dev_F = evaluate_F(config, model, dev_iter)
                ###########org
                if dev_F > dev_best_F:
                    dev_best_F = dev_F
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                ############ change
                # torch.save(model.state_dict(), config.save_path)
                # improve = '*'
                # last_improve = 0


                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2}, Entity Train Loss: {2:>5.2}, Context Train Loss: {3:>5.2}, Train Acc: {4:>6.2%},  Val Loss: {5:>5.2},  Val Acc: {6:>6.2%},  Time: {7} {8}'
                print(msg.format(total_batch, loss.item(), loss_entity, loss_context, train_acc, dev_loss, dev_acc, time_dif, improve))
                print("Precision, Recall and F1-Score...")
                print(dev_P, dev_R, dev_F)

                test(config, model, test_iter)

                model.train()

            total_batch += 1
            ################org
            if total_batch - last_improve > config.require_improvement:
                
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
    test(config, model, test_iter)


def test(config, model, test_iter):
    # test
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_P, test_R, test_F = evaluate_F(config, model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_P, test_R, test_F)

    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for (test_entity, test_context, labels) in data_iter:
            outputs, _, _ = model(test_entity, test_context)
            loss = torch.nn.functional.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)


    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        # report = metrics.classification_report(labels_all, predict_all)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter)


def only_test(config, model, test_iter):
    # test
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_P, test_R, test_F = only_evaluate_F(config, model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_P, test_R, test_F)

    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


def only_evaluate_F(config, model, data_iter, test=False):
    fw = open(Results_write_path,"w",encoding="utf-8")
    id_to_label = {0:'PER',1:'LOC',2:'ORG',3:'GPE',4:'O'}
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        count = 0
        for (test_entity, test_context, labels) in data_iter:
            count += 1
            if count % 1000 == 0:
                print("have processed %d\n"%count)
            outputs, _, _ = model(test_entity, test_context)
            loss = torch.nn.functional.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    # print("predict_all=====================")
    # print(predict_all)
    predict_all_2_label = []
    for id_label in predict_all:
        predict_all_2_label.append(str(id_to_label[id_label]))

    for id_label in predict_all:
        fw.write(str(id_to_label[id_label]))
        fw.write('\n')
    fw.close()

    fr_seg_path = Seg_path_test
    print("======================")
    print(predict_all_2_label)
    print(len(predict_all_2_label))
    no_nested_list = no_nested_process(fr_seg_path, predict_all_2_label)
    fr_pos_path = Postive_path_test
    # print("no_nested_list=============")
    # print(no_nested_list[0:5])
    P, R, F = compute_P_R_F(fr_pos_path, no_nested_list)


    acc = metrics.accuracy_score(labels_all, predict_all)
    # if test:
    #     report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
    #     # report = metrics.classification_report(labels_all, predict_all)
    #     confusion = metrics.confusion_matrix(labels_all, predict_all)
    #     return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter), P, R, F


def evaluate_F(config, model, data_iter, test=False):
    id_to_label = {0: 'PER', 1: 'LOC', 2: 'ORG', 3: 'GPE', 4: 'O'}
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for (test_entity, test_context, labels) in data_iter:
            outputs,_, _ = model(test_entity, test_context)
            loss = torch.nn.functional.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    predict_all_2_label = []
    for id_label in predict_all:
        predict_all_2_label.append(str(id_to_label[id_label]))

    if test:
        fr_seg_path = Seg_path_test
        fr_pos_path = Postive_path_test
    else:
        fr_seg_path = Seg_path
        fr_pos_path = Postive_path
    no_nested_list = no_nested_process(fr_seg_path, predict_all_2_label)
    P, R, F = compute_P_R_F(fr_pos_path, no_nested_list)
    acc = metrics.accuracy_score(labels_all, predict_all)
    # if test:
    #     report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
    #     # report = metrics.classification_report(labels_all, predict_all)
    #     confusion = metrics.confusion_matrix(labels_all, predict_all)
    #     return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter), P, R, F