If you has any question, please email to hunyun2016@iscas.ac.cn

# Decoupled_NER

We will update the code, data and readme soon. The temporary code can be seen in https://github.com/huyun-cs/Decoupled_NER/blob/main/1240_file_SupMat__Software.zip 

If you have read the code and need data, please send an email to huyuniot@163.com .

## Train:
nohup python -u run_3_loss_share.py --model bert_3_loss_share >ner_train_dropout.log 2>&1 &

## Test:
nohup python -u eval_3_loss.py --model bert_3_loss_share >ner_test_dropout.log 2>&1 &

## dataset:
OntoNotes： https://catalog.ldc.upenn.edu/LDC2011T03

MSRA: http://download.fastnlp.top/dataset/MSRA_NER.zip

Weibo : https://github.com/hltcoe/golden-horse

## Dictionary:
from: https://shurufa.baidu.com/dict

from: https://pinyin.sogou.com/dict/

## Distantly supervised data:
from: http://paper.people.com.cn

from: https://weibo.com/


## Detail
We will show the detail of the process.

### Annotation data

### Pretained data

### Pretrain process

put data in the ./deNER
