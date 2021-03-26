# Attention is all you need: A Pytorch Implementation

This is a PyTorch implementation of the Transformer model in "[Attention is All You Need](https://arxiv.org/abs/1706.03762)" (Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin, arxiv, 2017). 


A novel sequence to sequence framework utilizes the **self-attention mechanism**, instead of Convolution operation or Recurrent structure, and achieve the state-of-the-art performance on **WMT 2014 English-to-German translation task**. (2017/06/12)

> The official Tensorflow Implementation can be found in: [tensorflow/tensor2tensor](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py).

> To learn more about self-attention mechanism, you could read "[A Structured Self-attentive Sentence Embedding](https://arxiv.org/abs/1703.03130)".

<p align="center">
<img src="http://imgur.com/1krF2R6.png" width="250">
</p>


The project support training and translation with trained model now.

Note that this project is still a work in progress.

**BPE related parts are not yet fully tested.**


If there is any suggestion or error, feel free to fire an issue to let me know. :)
# Env
```python
dill==0.3.3
python==3.8.8
pytorch==1.8.0
spacy==3.0.3
torchtext==0.6.0
tqdm==4.56.0
```

# Usage

## WMT'16 Multimodal Translation: de-en

An example of training for the WMT'16 Multimodal Translation task (http://www.statmt.org/wmt16/multimodal-task.html).

### 0) Download the spacy language model.(https://spacy.io/usage)
```bash
conda install -c conda-forge spacy
python -m spacy download zh_core_web_sm / zh_core_web_trf
python -m spacy download en_core_web_sm / en_core_web_trf
python -m spacy download de_core_news_sm / de_dep_news_trf
```

### 1) Preprocess the data with torchtext and spacy.(main_wo_bpe())
```bash
python preprocess.py -lang_src de -lang_trg en -save_data m30k_deen_shr.pkl
```

### 2) Train the model
```bash
python train.py -data_pkl m30k_de_en_shr.pkl -proj_share_weight -label_smoothing -output_dir output -use_tb -b 256 -warmup 4000 -d_model 512 -d_inner_hid 1024 -d_k 32 -d_v 32 -n_head 8 -n_layers 6 -dropout 0.2
```

### 3) Test the model
```bash
python translate.py -data_pkl m30k_deen_shr.pkl -model trained.chkpt -output prediction.txt
```

## [(WIP)] WMT'17 Multimodal Translation: de-en w/ BPE 
### 1) Download and preprocess the data with bpe:

> Since the interfaces is not unified, you need to switch the main function call from `main_wo_bpe` to `main`.

```bash
python preprocess.py -raw_dir /tmp/raw_deen -data_dir ./bpe_deen -save_data bpe_vocab.pkl -codes codes.txt -prefix deen
```

### 2) Train the model
```bash
python train.py -data_pkl ./bpe_deen/bpe_vocab.pkl -train_path ./bpe_deen/deen-train -val_path ./bpe_deen/deen-val -log deen_bpe -embs_share_weight -proj_share_weight -label_smoothing -output_dir output -b 256 -warmup 128000 -epoch 400
```

### 3) Test the model (not ready)
- TODO:
	- Load vocabulary.
	- Perform decoding after the translation.
---

## WMT'17 Multimodal Translation: en-zh

An example of training for the WMT'17 Multimodal Translation task (http://data.statmt.org/wmt17/translation-task/training-parallel-nc-v12.tgz).

### 0) Download the language dateset.(preprocess.py-main())
```bash
_TRAIN_DATA_SOURCES = [
    {"url": "http://data.statmt.org/wmt17/translation-task/" \
             "training-parallel-nc-v12.tgz",
     "trg": "news-commentary-v12.zh-en.en",
     "src": "news-commentary-v12.zh-en.zh"}
```
可以借助preprocess.py-main()下载中英翻译数据集，也可以自己下载解压

### 1) Preprocess the data with torchtext and spacy.(create_zh_en())
```bash
python preprocess.py -lang_src en -lang_trg zh -save_data test_en_zh.pkl
```

### 2) Train the model
```bash
python train.py -data_pkl test_en_zh.pkl -proj_share_weight -label_smoothing -output_dir output
```

# Performance
## Training

<p align="center">
<img src="https://i.imgur.com/S2EVtJx.png" width="400">
<img src="https://i.imgur.com/IZQmUKO.png" width="400">
</p>

- Parameter settings:
  - batch size 256 
  - warmup step 4000 
  - epoch 200 
  - lr_mul 0.5
  - label smoothing 
  - do not apply BPE and shared vocabulary
  - target embedding / pre-softmax linear layer weight sharing. 
 
  
## Testing 
- coming soon.
---
# TODO
  - Evaluation on the generated text.
  - Attention weight plot.
---
# Acknowledgement
- The byte pair encoding parts are borrowed from [subword-nmt](https://github.com/rsennrich/subword-nmt/).
- The project structure, some scripts and the dataset preprocessing steps are heavily borrowed from [OpenNMT/OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py).
- Thanks for the suggestions from @srush, @iamalbert, @Zessay, @JulesGM, @ZiJianZhao, and @huanghoujing.

# Tensorboard[可视化](https://www.jianshu.com/p/46eb3004beca)
### 1)连接服务器虚拟python环境
```bash
启动pycharm-terminal
tensorboard --logdir output/tensorboard
```

# Screen后台运行
```bash
screen -ls 显示
screen -S session 创建
screen -r -S session 连接
screen -D -r -S session 抢占
```
