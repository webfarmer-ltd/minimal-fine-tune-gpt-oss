# minimal-fine-tune-gpt-oss repository
GPT OSS の超お手軽お試しコード

コード作成にあたり、こちらの記事を参考にさせてもらいました。
https://qiita.com/Maki-HamarukiLab/items/5ab44bc3f1836eca4bd1

## 環境構築
use docker.
```sh
bash DockerRun.sh
```

## gpt-ossモデルのdownload
```sh
python3  fine_tune_gpt_oss.py --process_name download_models
```

## gpt-oss 推論のみ実施
```sh
python3  fine_tune_gpt_oss.py --process_name predict
```

## gpt-oss LoRAの学習
```sh
python3  fine_tune_gpt_oss.py --process_name train_lora
```