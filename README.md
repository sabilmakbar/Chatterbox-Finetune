# 🎙️ Chatterbox Finetune (Custom)

Fine-tuning pipeline for Chatterbox-TTS, adapted from the original chatterbox-finetune repo.

This project focuses on training **T3 (text encoder)** and **S3 / Flow components** with custom datasets.

---

## 🚀 Overview

This repo provides:

* Fine-tuning scripts for Chatterbox TTS
* Support for Hugging Face datasets
* TensorBoard logging
* Training configs optimized for quick experimentation

Built on:

* chatterbox-finetune (base)
* Chatterbox-TTS (by Resemble AI)

---

## 📦 Installation

Using your local package setup:

```bash
pip install -r requirements.txt
```

Or with pyproject:

```bash
pip install -e .
```

---

## 🏋️ Fine-tuning

Run from `src/`:

```bash
cd src
```

### Example training command

```bash
#see finetune_s3gen.py for more details
python finetune_t3.py \
--output_dir ./checkpoints/chatterbox_finetuned \
--model_name_or_path ResembleAI/chatterbox \
--dataset_name $YOUR_REMOTE_HF_DATASET \
--train_split_name train \
--eval_split_size 0.0002 \
--num_train_epochs 1 \
--per_device_train_batch_size 4 \
--gradient_accumulation_steps 2 \
--learning_rate 5e-5 \
--warmup_steps 100 \
--logging_steps 10 \
--eval_strategy steps \
--eval_steps 2000 \
--save_strategy steps \
--save_steps 4000 \
--save_total_limit 4 \
--fp16 True \
--report_to tensorboard \
--dataloader_num_workers 8 \
--do_train --do_eval \
--dataloader_pin_memory False \
--eval_on_start True \
# label_names is defined inside the script, not in HF dataset columns, see more
--label_names labels_speech \
--text_column_name $YOUR_REMOTE_HF_DATASET_COL
```

---

## 📊 Monitoring

Launch TensorBoard:

```bash
tensorboard --logdir ./checkpoints
```

---

## 🧠 Notes

* Designed for **Python 3.12**
* If using GPU → ensure CUDA-compatible PyTorch
* Dataset should follow expected format (audio + text)
* Adjust inference params (`cfg_weight`, `exaggeration`) for style

---

## 🔊 Inference (after training)

```python
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS

model = ChatterboxTTS.from_pretrained(device="cuda")

text = "Your custom fine-tuned voice here."
wav = model.generate(text)

ta.save("output.wav", wav, model.sr)
```

---

## 🧩 Project Structure

```
.
├── src/
│   ├── finetune_t3.py
│   └── ...
├── checkpoints/
├── requirements.txt / pyproject.toml
└── README.md
```

---

## ⚠️ Gotchas

* numpy version conflicts → use Python 3.11
* Ensure dataset schema matches expected columns
* Large datasets → watch disk and RAM usage
* fp16 may not work on all GPUs

---

## Credits

* [Chatterbox Finetune base repo](https://github.com/stlohrey/chatterbox-finetuning)
* [Resemble AI Chatterbox-TTS](https://github.com/resemble-ai/chatterbox)

---

## ⚠️ Disclaimer

Use responsibly. Do not generate misleading or harmful synthetic audio.
