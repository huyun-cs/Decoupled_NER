

def compute_P_R_F(fr_pos_path, fr_test_list):
    gold_label = {}
    word_key_set = set()
    word_key_num = {}

    fr_pos = open(fr_pos_path, "r", encoding="utf-8")
    pos_list = fr_pos.readlines()

    for line in pos_list:
        word_label = line.strip().split("\t")
        word = word_label[0].strip()
        context = word_label[1].strip()
        if len(context.strip().split()) > 125:
            context = context[:-4]
        label = word_label[2]
        maskpos = word_label[-1]
        word_key = word + "\t" + context
        gold_label[word_key] = label
        word_key_set.add(word_key)
        if word_key in word_key_num:
            word_key_num[word_key] += 1
        else:
            word_key_num[word_key] = 1
    ALL_gold = 0
    for word in word_key_num.keys():
        ALL_gold += word_key_num[word]
    # ALL_gold = len(word_key_set)

    right_num = 0
    pred_num = 0
    bianjie = 0
    word_key_in_neg = set()
    i = 0

    pred_label = {}

    for line in fr_test_list:
        word_label = line.strip().split("\t")
        word = word_label[0].strip()
        context = word_label[1].strip()
        label = word_label[2]
        maskpos = word_label[-1]
        word_key = word + "\t" + context

        if label != 'O':
            pred_label[word_key] = label
            pred_num += 1
            if word_key in gold_label:
                if gold_label[word_key] == label:
                    right_num += word_key_num[word_key]
                    word_key_in_neg.add(word_key)
                else:
                    pass
                bianjie += 1
            
        i += 1

    print(right_num)
    print(ALL_gold)
    print(pred_num)
    try:
        P = right_num / pred_num
        R = right_num / ALL_gold
        F = 2 * P * R / (P + R)
    except:
        P = 0
        R = 0
        F = 0
    return P, R, F


def no_nested_process(fr_seg_path, pred_list):

    fr_seg = open(fr_seg_path,"r",encoding="utf-8")
    fr_seg_list = fr_seg.readlines()

    i = 0
    sentence_dict = {}
    sentence_dict_only_label = {}
    for line in fr_seg_list:
        if i % 5000 == 0:
            print("have processed %d \n" % i)
        if pred_list[i] != 'O':
            word_list = []
            word_label = line.strip().split("\t")
            word = word_label[0].strip()
            context = word_label[1].strip().split()
            context_str = word_label[1].strip()
            label = pred_list[i]
            start = int(word_label[-1])
            end = start + len(word.split())
            for char in context:
                if char == 'mask':
                    for entity_char in word.split():
                        word_list.append(entity_char)
                else:
                    word_list.append(char)
            word_str = "".join(word_list)
            if word_str in sentence_dict:
                label_list = sentence_dict[word_str]
                label_list_only_label = sentence_dict_only_label[word_str]
                label_list.append((start, end, word, context_str, label, str(start)))
                label_list_only_label.append((start, end, label))
                sentence_dict[word_str] = label_list
                sentence_dict_only_label[word_str] = label_list_only_label
            else:
                label_list = []
                label_list_only_label = []
                label_list.append((start, end, word, context_str, label, str(start)))
                label_list_only_label.append((start, end, label))
                sentence_dict[word_str] = label_list
                sentence_dict_only_label[word_str] = label_list_only_label
                # if i < 20:
                #     print("sentence_dict=================")
                #     print(word_str)
                #     print(label_list)

        i += 1

    # print("len_sentence_dict==============")
    # print(len(sentence_dict))

    count = 1

    no_nested_list = []
    aaa = 1
    for word_str in sentence_dict.keys():
        # aaa += 1
        # if aaa < 5:
        #     print("word_str++++++++++++++")
        #     print(word_str)
        if count % 1000 == 0:
            print("have write %d\n" % count)
        count += 1
        label_list = sentence_dict[word_str]
        label_list_only_label = sentence_dict_only_label[word_str]
        remove_set = set()
        # print(label_list)
        for i in range(len(label_list)):
            for j in range(len(label_list)):
                if i != j:
                    start_i = label_list[i][0]
                    end_i = label_list[i][1]
                    label_i = label_list[i][4]
                    start_j = label_list[j][0]
                    end_j = label_list[j][1]
                    label_j = label_list[j][4]

                    if (start_i <= start_j) and (end_i >= end_j):
                        if (start_i == start_j) and ((end_j, end_i, label_j) in label_list_only_label) and (
                            label_j == label_i):
                            remove_set.add(i)
                            continue
                        if (end_j == end_i) and ((start_i, start_j, label_j) in label_list_only_label) and (
                            label_j == label_i):
                            remove_set.add(i)
                            continue
                    if (start_i >= start_j) and (end_i <= end_j):
                        if (start_i == start_j) and ((end_i, end_j, label_j) in label_list_only_label) and (
                            label_j == label_i):
                            pass
                        elif (end_j == end_i) and ((start_j, start_i, label_j) in label_list_only_label) and (
                            label_j == label_i):
                            pass
                        else:
                            remove_set.add(i)
        # print(remove_set)


        # print("remove set=============")
        # bbb = 1
        # for j in remove_set:
        #     bbb += 1
        #     if bbb < 5:
        #         print(j)

        for i in range(len(label_list)):
            if i not in remove_set:
                no_nested_list.append(label_list[i][2] + "\t" + label_list[i][3] + "\t" + \
                         label_list[i][4] + "\t" + label_list[i][5] + "\n")

    return no_nested_list

