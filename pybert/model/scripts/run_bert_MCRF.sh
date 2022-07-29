CURRENT_DIR=`pwd`
export BERT_BASE_DIR=$CURRENT_DIR/prev_trained_model/bert-base
export DATA_DIR=$CURRENT_DIR/datasets
export OUTPUR_DIR=$CURRENT_DIR/outputs
TASK_NAME="mcrf"
#
python MCRF_trainer.py \
  --model_type=bert \
  --model_name_or_path=$BERT_BASE_DIR \
  --task_name=$TASK_NAME \
  --do_train \
  --do_predict \
  --do_lower_case \
  --data_dir=$DATA_DIR/${TASK_NAME}/ \
  --train_max_seq_length=512 \
  --eval_max_seq_length=512 \
  --per_gpu_train_batch_size=8 \
  --per_gpu_eval_batch_size=8 \
  --learning_rate=5e-7 \
  --crf_learning_rate=7e-4 \
  --num_train_epochs=2.0 \
  --logging_steps=-1 \
  --save_steps=-1 \
  --output_dir=$OUTPUR_DIR/${TASK_NAME}_output/ \
  --overwrite_output_dir \
  --seed=42 \
  --gradient_accumulation_steps=4
