from utils import *
import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, T5ForConditionalGeneration

device_id = 2
device = torch.device(f"cuda:{device_id}")
torch.cuda.set_device(device_id)
def main(src_path, tgt_path, model_id = "declare-lab/flan-alpaca-large"):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, device_map=device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id, device_map=device)
    with open(src_path, 'r') as f:
        lines = f.read().split('\n\n')
    
    line = lines[0]
    input_ids = tokenizer(line, return_tensors='pt').input_ids.to(device)
    output = model.get_encoder()(input_ids)
    breakpoint()

     
if __name__ == '__main__':
    # Load data
    src_path = 'source/reference.txt'
    tgt_path = 'target/embeds.csv'  
    
    main(src_path, tgt_path)