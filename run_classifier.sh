#!/bin/sh

# 分类任务
#export CUDA_VISIBLE_DEVICES=""
#export BERT_BASE_DIR=/home/czm/Public/bert/pretrained_models/uncased_L-12_H-768_A-12
#export BERT_BASE_DIR=/home/czm/Public/bert/pretrained_models/uncased_L-8_H-512_A-8
export BERT_BASE_DIR=/home/czm/Public/bert/pretrained_models/uncased_L-2_H-128_A-2
export GLUE_DIR=/home/czm/Public/bert/datasets/glue_data

python run_classifier.py \
  --task_name=MRPC \
  --do_train=true \
  --do_eval=true \
  --data_dir=$GLUE_DIR/MRPC \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --output_dir=./record/mrpc_output



