# FAN: Fourier Analysis Networks
[**Paper**](https://arxiv.org/abs/2410.02675)

![The performance of different neural networks within and outside the domain of their training data for the sine function, where x is a scalar variable.](./img/intro.pdf)

## Periodicity Modeling
```shell
cd Periodicity_Modeling
bash ./run.sh
```

## Sentiment Analysis
The data can be automatically downloaded using the Huggingface Datasets `load_dataset` function in the `./Sentiment_Analysis/get_dataloader.py`. 

```shell
cd Sentiment_Analysis
bash scripts/Trans_with_FAN/train_ours.sh
bash scripts/Trans_with_FAN/test_ours.sh
```

## Timeseries Forecasting
You can obtain data from [Google Drive](https://drive.google.com/drive/folders/1v1uLx5zhGaNAOTIqHLHYMXtA-XFrKTxS?usp=sharing). All the datasets are well pre-processed and can be used easily.

```shell
cd Timeseries_Forecasting
bash scripts/Weather_script/Modified_Transformer.sh 
```

## Symbolic Formula Representation
```shell
cd Symbolic_Formula_Representation
python gen_dataset.py
bash run_train_fan.sh
```

## Citation
```
@article{dong2024fan,
  title={FAN: Fourier Analysis Networks},
  author={Yihong Dong and Ge Li and Yongding Tao and Xue Jiang and Kechi Zhang and Jia Li and Jing Su and Jun Zhang and Jingjing Xu},
  journal={arXiv preprint arXiv:2410.02675},
  year={2024}
}
```
