from threading import Thread
from rank_bm25 import BM25Okapi
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TextIteratorStreamer
from utils import *

model_id = "declare-lab/flan-alpaca-large"
# torch_device = "cuda" if torch.cuda.is_available() else "cpu"
# print("Running on device:", torch_device)
print("CPU threads:", torch.get_num_threads())
torch.backends.cudnn.benchmark = True
device_id = 2
device = torch.device(f"cuda:{device_id}")
torch.cuda.set_device(device_id)

# model = AutoModelForSeq2SeqLM.from_pretrained(model_id, load_in_8bit=True, device_map=device)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id, device_map=device)
tokenizer = AutoTokenizer.from_pretrained(model_id)
f = open("source/reference.txt", "r").read().split('\n\n')
tokenized_corpus = [doc.split(" ") for doc in f]
bm25 = BM25Okapi(tokenized_corpus)

def myBM25(user_text):
    tokenized_query = user_text.split(" ")
    contexts = bm25.get_top_n(tokenized_query, tokenized_corpus, n=3)
    prompt = "Answer the question below, you can refer to but NOT limited to the contexts \n\nContext: \n"
    for context in contexts:
        prompt += ' '.join(context) + "\n\n###\n\n"
    prompt += "---\n\nQuestion: " + user_text + "\nAnswer:"
    return prompt

def run_generation(model_name, user_text, top_p, temperature, top_k, max_new_tokens):
    # Get the model and tokenizer, and tokenize the user text.
    # print('Model name', model_name)
    if model_name == 1:
        # model = Prompt_GPT()
        # return model.get_chatGPT(user_text, top_p, temperature, top_k, max_new_tokens)
        # t = Thread(target=chatGPT, args=(user_text, top_p, temperature, top_k, max_new_tokens))
        # streamer = TextIteratorStreamer(None, timeout=10., skip_prompt=True, skip_special_tokens=True)
        # t = Thread(target=simple_return, args=())
        # t.start()
        model_output = "check"
        yield model_output
        return model_output
        
    else:
        contexted_user_text = myBM25(user_text)
        model_inputs = tokenizer([contexted_user_text], return_tensors="pt").to(device)

        # Start generation on a separate thread, so that we don't block the UI. The text is pulled from the streamer
        # in the main thread. Adds timeout to the streamer to handle exceptions in the generation thread.
        streamer = TextIteratorStreamer(tokenizer, timeout=10., skip_prompt=True, skip_special_tokens=True)
        generate_kwargs = dict(
            model_inputs,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=top_p,
            temperature=float(temperature),
            top_k=top_k
        )
        t = Thread(target=model.generate, kwargs=generate_kwargs)
        t.start()

        # Pull the generated text from the streamer, and update the model output.
        model_output = ""
        for new_text in streamer:
            model_output += new_text
            yield model_output
        return model_output

def reset_textbox():
    return gr.update(value='')

with gr.Blocks() as demo:
    duplicate_link = "https://huggingface.co/spaces/joaogante/transformers_streaming?duplicate=true"
    gr.Markdown(
        "# 🤗 Transformers 🔥Streaming🔥 on Gradio\n"
        "This demo showcases the use of the "
        "Transformers with Gradio to generate text in real-time. It uses "
        f"[{model_id}](https://huggingface.co/{model_id}).\n\n"
    )

    with gr.Row():
        with gr.Column(scale=4):
            user_text = gr.Textbox(
                placeholder="Ask Solar PV-related queries here",
                label="User input",
                lines = 2
            )
            model_output = gr.Textbox(label="Model output", lines=8, interactive=False)
            context_output = gr.Textbox(
                show_label=True,
                label ="Contextual information",
                lines = 6
                )
            button_submit = gr.Button(value="Submit")
            

        with gr.Column(scale=1):

            model_name = gr.Radio(["Flan-Alpaca", "chatGPT-3.5"], 
                                  value="Flan-Alpaca", label="Model",
                                  type = 'index')
            
            max_new_tokens = gr.Slider(
                minimum=1, maximum=1000, value=250, step=1, interactive=True, label="Max New Tokens",
            )
            top_p = gr.Slider(
                minimum=0.05, maximum=1.0, value=0.95, step=0.05, interactive=True, label="Top-p (nucleus sampling)",
            )
            top_k = gr.Slider(
                minimum=1, maximum=50, value=50, step=1, interactive=True, label="Top-k",
            )
            temperature = gr.Slider(
                minimum=0.1, maximum=5.0, value=0.3, step=0.1, interactive=True, label="Temperature",
            )
            
    user_text.submit(run_generation, [model_name, user_text, top_p, temperature, top_k, max_new_tokens], model_output)
    user_text.submit(myBM25, [user_text], context_output)
    button_submit.click(run_generation, [model_name, user_text, top_p, temperature, top_k, max_new_tokens], model_output)
    button_submit.click(myBM25, [user_text], context_output)
    demo.queue(max_size=32).launch(share=False)