from threading import Thread

import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TextIteratorStreamer
# from utils import *
from create_context import *
from openai import OpenAI

model_id = "declare-lab/flan-alpaca-large"
client = OpenAI(api_key = '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')
print("CPU threads:", torch.get_num_threads())
torch.backends.cudnn.benchmark = True
device_id = 2
device = torch.device(f"cuda:{device_id}")
torch.cuda.set_device(device_id)

# model = AutoModelForSeq2SeqLM.from_pretrained(model_id, load_in_8bit=True, device_map=device)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id, device_map=device)
tokenizer = AutoTokenizer.from_pretrained(model_id)
context_df = pd.read_csv('target/chunk_embeddings.csv', index_col=0)
context_df = context_df[context_df['text']!='\n']
def run_generation(model_name, user_text, top_p, temperature, top_k, max_new_tokens):
    # Get the model and tokenizer, and tokenize the user text.
    # print('Model name', model_name)
    if model_name == 1:
        contexted_user_text = search_contexts(context_df, user_text, n=3)
        
        model_output = client.chat.completions.create(
            model = "gpt-3.5-turbo",
            messages = [{'role':'user', 'content':contexted_user_text}]
        ).choices[0].message.content
        # model_output = 'test'
        print(contexted_user_text)
        yield model_output
        return model_output
        
    else:
        contexted_user_text = search_contexts(context_df, user_text, n=3)
        
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
    # user_text.submit(search_contexts, [context_df, user_text, 3], context_output)
    button_submit.click(run_generation, [model_name, user_text, top_p, temperature, top_k, max_new_tokens], model_output)
    # button_submit.click(search_contexts, [context_df, user_text, 3], context_output)
    demo.queue(max_size=32).launch(share=True)