from googlesearch import search
import requests
from bs4 import BeautifulSoup
# "How are the HDB blocks selected for solar PV installations? "

from search_utils import find_most_relevant_paragraph, rank_sentence_transformer
f = open('contexts.txt', 'w')
query = "What is the type of solar panel technology used?"
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36',
}
hit = 0
for url in search(query, num_results=10):
    # Send a HTTP request to the URL
    print(url)
    if hit > 5:
        f.close()
        break
    response = requests.get(url, headers=headers)
    # Check if the request was successful
    if response.status_code == 200:
        # Parse the content of the response using BeautifulSoup
        try: 
            soup = BeautifulSoup(response.content, 'html.parser')
        except:
            continue
        # Example: Extract all the paragraph texts
        paragraphs = soup.find_all('p')
        if len(paragraphs) == 0:
            continue
        # for paragraph in paragraphs:
        #     print(paragraph.text)
        cleaned_paragraphs = []
        for p in paragraphs:
            p = p.text.strip()
            if len(p) > 128:
                cleaned_paragraphs.append(p)
        # bm25 = BM25Okapi(cleaned_paragraphs)
        # contexts = bm25.get_top_n(query, cleaned_paragraphs, n=5)
        # contexts = find_most_relevant_paragraph(cleaned_paragraphs, query)
        contexts = rank_sentence_transformer(cleaned_paragraphs, query)
        f.write('Source: ' + url + '\n')
        f.write('\n\n'.join(contexts))
        f.write('----------------------------\n\n')
        hit += 1
    else:
        print(f"Failed to retrieve content from {url}, status code: {response.status_code}")

