import pandas as pd
import numpy as np
from openai import OpenAI
from tqdm.auto import tqdm
client = OpenAI(api_key = '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')

def cosine_similarity(a, b):
    if type(a) == str:
        a = a[1:-1].split(', ')
        a = [float(i) for i in a]
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_embedding(text, model="text-embedding-ada-002"):
   text = str(text).replace("\n", " ")
   print(text)
   return client.embeddings.create(input = [text], model=model).data[0].embedding

def search_contexts(df, text, n=1):
    text_embeds = get_embedding(text)
    
    df['similarity'] = df['embedding'].apply(lambda x: cosine_similarity(x, text_embeds))
    res = df.sort_values(by=['similarity'], ascending=False).head(n)
    if n==1:
        context = res.text.tolist()[0]
    else:
        context = ''.join(res.text.tolist())
    query = f"Answer the question below, you can refer the contexts provided if you think it is helpful, otherwise you just refer to your knowledge base. Context \n{context}"
    query += "\nQuestion: " + text
    return query

# f = open("source/chunks.txt", "r")
# lines = f.readlines()
# df = pd.DataFrame(columns=['id', 'text', 'embedding'])
# df = df.set_index('id')
# bar = tqdm(total=len(lines))
# for i, line in enumerate(lines):
#     embeds = get_embedding(line)
#     df.loc[i] = [line, embeds]
#     bar.update(1)
#     ## change df embedding data type to list
# df.to_csv('target/chunk_embeddings.csv') 
if __name__ == '__main__':
    df = pd.read_csv('target/chunk_embeddings.csv', index_col=0)
    df = df[df['text']!='\n']
    search_contexts(df, 'I am a student', n=1)