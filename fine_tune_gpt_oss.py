from unsloth import FastLanguageModel
import torch
import argparse
from transformers import TextStreamer
import os
os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
from torch.utils.data import DataLoader

os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"
import torch.nn.functional as F
import prompt.prompt as train_prompt

def parser():
    parser = argparse.ArgumentParser(description='lllm_args')
    parser.add_argument('--process_name', '-pn', type=str, default='load_pdf', help='process name')
    parser.add_argument('--output_dir', '-od', type=str, default='../debug/', help='path to output directory')
    parser.add_argument('--input_dir', '-id', type=str, default='src_nitto_csv', help='path to input directory')
    parser.add_argument('--text_type', '-tt', type=str, default='ehime', help='text type used for training')
    return parser.parse_args()

def download_models():
    max_seq_length = 4096
    dtype = None

    model, tokenizer = FastLanguageModel.from_pretrained( # これを実行するとgpt-ossモデルがdownloadされる
        model_name = "unsloth/gpt-oss-20b", # ここに無ければ、downlaodして保存される
        dtype = dtype,
        max_seq_length = max_seq_length,
        load_in_4bit = True,  # downloadされるモデルはfp16だが、ここで4bitに替えられる
        full_finetuning = False,
        # cache_dir = "./hf_cache",
    )

def predict(args):
    max_seq_length = 4096
    dtype = None
    # modelのload
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/gpt-oss-20b", # ここにあればloadされる
        dtype = dtype,
        max_seq_length = max_seq_length,
        load_in_4bit = True,  # 4bit
        # load_in_8bit = True,  # 8bit
        full_finetuning = False,
    )
    print("tokenizer ", tokenizer)

    messages = [
    {"role": "user", "content": "{}".format(train_prompt.get_train_text(args.text_type))},
    ]

    # 推論努力レベルを設定（low/medium/high）
    inputs = tokenizer.apply_chat_template( # text -> tensor[[1, 1433, 29871, 510, 13, 12345, ...]]
        messages,
        add_generation_prompt = True,
        return_tensors = "pt",
        return_dict = True,
        reasoning_effort = "medium",  # ここで設定！
    ).to(model.device)

    output_ids = model.generate(
                                **inputs,
                                max_new_tokens = 1024,
                                streamer = TextStreamer(tokenizer))
    print("output_ids", output_ids)
    print(tokenizer.decode(output_ids[0]))


def get_lora_norms(model):
    stats = {}
    for name, p in model.named_parameters():
        if "lora" in name and p.requires_grad:
            stats[name] = p.detach().norm().item()
    return stats

def get_lora_deltas(model, initial_lora):
    deltas = {}
    for name, p in model.named_parameters():
        if name in initial_lora:
            delta = (p.detach() - initial_lora[name])
            deltas[name] = delta.norm().item()
    return deltas


def train_simple_lora(args):
    max_seq_length = 1024
    dtype = torch.bfloat16
    # dtype = None

    # モデルロード（4bit量子化）
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/gpt-oss-20b",
        dtype = dtype,
        max_seq_length = max_seq_length,
        load_in_4bit = True,
        full_finetuning = False,
    )

    # LoRA層を追加
    model = FastLanguageModel.get_peft_model(
        model,
        # r = 8,
        r = 32,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 16,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
    )
    # 念のためモデル全体を bf16 へ（LoRA含む）
    model = model.to(dtype)

    # train_text = "ユーザー: 2+2は？\nアシスタント: 4です。"
    train_text = train_prompt.get_train_text(args.text_type)

    # tokenizerで直接変換
    inputs = tokenizer( # 
        train_text,
        return_tensors="pt",
        max_length=1024,
        truncation=True,
    ).to(model.device)

    # 教師データはシフトしたinput_ids（自己回帰モデル）
    labels = inputs["input_ids"].clone()

    # DataLoaderを構築
    dataset = [(inputs["input_ids"].squeeze(0), 
                inputs["attention_mask"].squeeze(0), labels.squeeze(0))]
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    # ==== 学習設定 ====
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model.train()

    # 学習開始前にloraの学習parameter名を記録
    initial_lora = {
        name: p.detach().clone()
        for name, p in model.named_parameters()
        if "lora" in name and p.requires_grad
    }

    lora_norm_history = []  # 各 epoch ごとに dict を append

    print("=== LoRAによる学習開始 ===")
    for epoch in range(30):  # 簡易的に1エポックのみ
        for input_ids, attention_mask, labels in loader:
            outputs = model(
                input_ids=input_ids.to(model.device),
                attention_mask=attention_mask.to(model.device),
                labels=labels.to(model.device),
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print(f"epoch:{epoch+1}. loss = {loss.item():.4f}")
            lora_norm_history.append(get_lora_norms(model))
    print("=== 学習完了 ===")

    # ==== 学習済みモデルで推論 ====
    model.eval()
    messages = [
        # {"role": "user", "content": "2+2は？"},
        {"role": "user", "content": "2025年9月以降の日本の首都は？"},
    ]
    test_inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)

    streamer = TextStreamer(tokenizer)
    output_ids = model.generate(**test_inputs, max_new_tokens=64, streamer=streamer)
    print("\n--- 出力 ---")
    print(tokenizer.decode(output_ids[0], skip_special_tokens=True))



if __name__ == "__main__":
    """
    1) download gpt-oss model
    python3  fine_tune_gpt_oss.py --process_name download_models

    2) predict model
    python3 fine_tune_gpt_oss.py --process_name predict --text_type math_tanzent

    3) train lora
    python3 fine_tune_gpt_oss.py --process_name train_lora --text_type simultaneous equations
    """
    args = parser()
    if args.process_name == 'download_models':
        download_models()
    elif args.process_name == 'train_lora':
        train_simple_lora(args)
    elif args.process_name == 'predict':
        predict(args)
    else:
        print("nothing done.")