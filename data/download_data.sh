#!/bin/bash
echo "downloading datasets"

## ** SemEval Dataset **
wget -O semeval.zip https://www.dropbox.com/s/b5c3nk54pr85836/semeval.zip?dl=0;
unzip semeval.zip

# pre-computed BERT embeddings 
wget -O semeval_bert.zip https://www.dropbox.com/s/pwgmzrzid8774de/semeval_bert.zip?dl=0;
mkdir pretrained_bert
unzip semeval_bert.zip -d pretrained_bert/semeval


## ** OPOSUM Dataset **
# To download our pre-processed version, uncomment the next lines of code (datafiles are large)
#  wget -O oposum.zip https://www.dropbox.com/s/u1hmjsuwy8zuhfj/oposum.zip?dl=0;
#  unzip oposum.zip -d oposum

# pre-computed BERT embeddings
#  wget -O bags_and_cases.zip https://www.dropbox.com/s/teoq57uhve0jszi/bags_and_cases.zip?dl=0;
#  wget -O bluetooth.zip https://www.dropbox.com/s/9qg8ikrxkq7v3aq/bluetooth.zip?dl=0;
#  wget -O boots.zip https://www.dropbox.com/s/u3v5f7fyhxnvkv8/boots.zip?dl=0;
#  wget -O keyboards.zip https://www.dropbox.com/s/pcj5auzxtb094pm/keyboards.zip?dl=0;
#  wget -O tv.zip https://www.dropbox.com/s/0qa7nwbrs19etyl/tv.zip?dl=0;
#  wget -O vacuums.zip https://www.dropbox.com/s/48jk5vxpg53s5ny/vacuums.zip?dl=0;

#  unzip bags_and_cases.zip -d pretrained_bert/oposum
#  unzip bluetooth.zip -d pretrained_bert/oposum
#  unzip boots.zip -d pretrained_bert/oposum
#  unzip keyboards.zip -d pretrained_bert/oposum
#  unzip tv.zip -d pretrained_bert/oposum
#  unzip vacuums.zip -d pretrained_bert/oposum

