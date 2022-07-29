## Improving Multi-label Malevolence Detection in Dialogues through Multi-faceted Label Correlation Enhancement

## Dependencies

- csv
- tqdm
- numpy
- pickle
- scikit-learn
- PyTorch1.5+
- matplotlib
- pandas
- transformers=4.5.1

## How to use the code

### data

#### data preprocessing 

We use MDMDdata which is uploaded to openview.net

1. Put the dataset file in the `./pybert/dataset/data_transform/data_input/multi-label-n`

2. Use `./pybert/dataset/data_transform/data_preprocessiong_n.py` to split the data, the splited data will saved in `./pybert/dataset/data_transform/data_set/`

3. `cd pybert/dataset/data_transform`

4. Run `python data_preprocessiong_n` to preprocess the data. 

### bert

Bert model is part of  the bert-MCRF model and is a baseline model. 

you need to download pretrained bert model firstly.

<div class="note info"><p> BERT:  bert-base-uncased</p></div>

1. Download the Bert pretrained model from [here](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-pytorch_model.bin) 
2. Download the Bert config file from [here](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-config.json) 
3. Download the Bert vocab file from [here](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt) 
4. Rename:

    - `bert-base-uncased-pytorch_model.bin` to `pytorch_model.bin`
    - `bert-base-uncased-config.json` to `config.json`
    - `bert-base-uncased-vocab.txt` to `bert_vocab.txt`
5. Place `model` ,`config` and `vocab` file into  the `/pybert/pretrain/bert/base-uncased` directory.
6. `pip install pytorch-transformers` from [github](https://github.com/huggingface/pytorch-transformers).
7. We use MDMDdata, which is uploaded to `openview.net`. 
8. Modify configuration information in `pybert/configs/basic_config.py`(the path of data,...).
9. Run `python run_bert.py --do_data` to transfer the splited data to formulated data which will be save in  `./pybert/dataset/data_transform/data_set/`
10. Run `python run_bert.py --do_train --save_best --do_lower_case` to fine tuning bert model.
11. Run `run_bert.py --do_test --do_lower_case` to predict new data. 
12. After training the bert model, we move the the trained bert model from `./pybert/output/checkpoints/bert` to `pybert/model/prev_trained_model/bert-base` and continue the save the Bert-MCRF model. 

### bert-MCRF

Run `cd pybert/model` to prepare to train the bert-MCRF model. 

1. Set `TASK_NAME="mcrf"` in `./pybert/model/models/scripts/run_bert_MCRF.sh`
2. Set `--do_train` 
3. run `bash scripts/run_bert_MCRF.sh` with 2 GPUs to train the bert-MCRF model. 
4. Set `--predict` 
5. run `bash scripts/run_bert_MCRF.sh` with 2 GPUs to train the bert-MCRF model. 

### bert-CRF

Bert-CRF is a baseline model. 

Run `cd pybert/model` to prepare to train the bert-CRF model. 

1. Set `TASK_NAME="crf"` in `./pybert/model/models/scripts/run_bert_CRF.sh`
2. Set `--do_train` 
3. run `bash scripts/run_bert_CRF.sh` with 2 GPUs to train the bert-CRF model. 
4. Set `--predict` 
5. run `bash scripts/run_bert_CRF.sh` with 2 GPUs to train the bert-CRF model. 

#### Ablation study

#### w/o LCC

1. Set `TASK_NAME="mcrfwolcc"` in `./pybert/model/scripts/run_bert_MCRF.sh`
2. In `./pybert/model/models/bert_MCRF.py` change the `from .layers.CRFKL import CRF` with `from .layers.CRFKLwoLCC import CRF`
3. run `bash scripts/run_bert_MCRF.sh` with 2 GPUs to train the bert-MCRF model. 
4. Set `--predict` 
5. run `bash scripts/run_bert_MCRF.sh` with 2 GPUs to train the bert-MCRF model. 

#### w/o LCT

1. Set `TASK_NAME="mcrfwolct"` in `./pybert/model/scripts/run_bert_MCRF.sh`
2. In `./pybert/model/models/bert_MCRF.py` delete `position_ids=position_ids` in line 78. 
3. Use line99 and delete line101 and line 102. 
4. run `bash scripts/run_bert_MCRF.sh` with 2 GPUs to train the bert-MCRF model. 
5. Set `--predict` 
6. run `bash scripts/run_bert_MCRF.sh` with 2 GPUs to train the bert-MCRF model. 

#### w/o LLCT

1. Set `TASK_NAME="mcrfwolct"` in `./pybert/model/scripts/run_bert_MCRF.sh
2. Use line99 and delete line101 and line 102. 
3. run `bash scripts/run_bert_MCRF.sh` with 2 GPUs to train the bert-MCRF model. 
4. Set `--predict` 
5. run `bash scripts/run_bert_MCRF.sh` with 2 GPUs to train the bert-MCRF model. 

#### w/o PLCT

1. Set `TASK_NAME="mcrfwolct"` in `./pybert/model/scripts/run_bert_MCRF.sh`
2. In `./pybert/model/models/bert_MCRF.py` delete `position_ids=position_ids` in line 78. 
3. run `bash scripts/run_bert_MCRF.sh` with 2 GPUs to train the bert-MCRF model. 
4. Set `--predict` 
5. run `bash scripts/run_bert_MCRF.sh` with 2 GPUs to train the bert-MCRF model. 


# Please refer to our paper:
@inproceedings{zhang2022improving,
  title={Improving Multi-label Malevolence Detection in Dialogues through Multi-faceted Label Correlation Enhancement},
  author={Zhang, Yangjun and Ren, Pengjie and Deng, Wentao and Chen, Zhumin and Rijke, Maarten},
  booktitle={Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={3543--3555},
  year={2022}
}


