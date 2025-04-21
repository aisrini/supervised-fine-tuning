# supervised-fine-tuning
# Fireworks.ai Fine‑Tuning Helper

A simple CLI tool to help you:

1. **Create** a new dataset record on Fireworks.ai  
2. **Upload** a local JSONL file to that dataset  
3. **Launch** a supervised fine‑tuning job  

This script keeps your API credentials out of source control (via `.env`), uses structured functions for each step, and provides clear logging and error handling.

---

## 🚀 Features

- Loads `ACCOUNT_ID` and `API_TOKEN` from environment variables  
- Modular functions for dataset creation, file upload, and fine‑tuning  
- CLI interface with `argparse`  
- Informative `logging` output  
- Automatic directory creation for downloaded/uploaded files  

---

## 🔧 Prerequisites

- Python 3.7 or higher  
- An active Fireworks.ai account  
- A [Personal Access Token](https://docs.fireworks.ai/authentication)  
- A local JSONL dataset file  

---

## 🛠 Installation

1. **Clone the repo**  
   ```bash
   git clone https://github.com/yourusername/fireworks-finetuning-helper.git
   cd fireworks-finetuning-helper
2. **Install dependencies**
  pip install -r requirements.txt
3. **Set up your environment variables**
   Copy the example file and fill in your credentials
   cp .env.example .env
   Edit .env and set:
   FIREWORKS_ACCOUNT_ID=your_account_id_here
   FIREWORKS_API_TOKEN=your_api_token_here

---

## ⚙️ Configuration
**.env**
FIREWORKS_ACCOUNT_ID=<ACCOUNT_ID>

FIREWORKS_API_TOKEN=<YOUR_FW_API_KEY>


---
**DATA_DIR**
- By default, your local JSONL will be uploaded from ./fireworks_datasets/; you can change this path in the script if needed.

## 💡 Usage
python fireworks_finetune.py \
  --dataset-id trader-poe-sample-data \
  --local-file ./fireworks_datasets/trader_poe_sample_data.jsonl \
  --display-name "Trader Poe's Fine Tuning" \
  --base-model accounts/<ACCOUNT_ID>/models/deepseek-r1-distill-llama-70b \
  --output-model accounts/<ACCOUNT_ID>/models/<OUTPUT_MODEL_NAME>


--dataset-id
A unique identifier for your dataset on Fireworks.ai.

--local-file
Path to the .jsonl file you want to upload.

--display-name
A human‑readable name for your fine‑tuning job.

---
## 📁 Project Structure

├── fireworks_finetune.py     # Main script

├── .env.example              # Example env file

├── requirements.txt          # Python dependencies

├── README.md                 # This file

└── fireworks_datasets/       # Local data directory

---
## 🤝 Contributing
- Fork the repo
- Create a feature branch (git checkout -b feature/foo)
- Commit your changes (git commit -am 'Add feature')
- Push to the branch (git push origin feature/foo)
- Open a Pull Request

---
## 📜 License
This project is licensed under the MIT License.

--base-model & --output-model
Full URIs of the base model and where you want your fine‑tuned model saved.

Made with ❤️ by Aishwarya Srinivasan


