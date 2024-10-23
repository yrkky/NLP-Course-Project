
import os
import zipfile
import requests
import nltk
import matplotlib.pyplot as plt
import fasttext as ft
import fasttext.util as ftutil
import numpy as np
import re
from nltk.tokenize import sent_tokenize, word_tokenize
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

################################################################################
################################################################################
################################################################################

def download_and_extract(url, zip_path, extract_path):
    # Check if the file already exists
    if os.path.exists(extract_path):
        print(f"File {extract_path} already exists.")
    else:
        print("Downloading and extracting the file...")
        # Download the file
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte

        with open(zip_path, 'wb') as file, tqdm(
            desc=zip_path,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(block_size):
                file.write(data)
                bar.update(len(data))

        # Create the data directory if it doesn't exist
        if not os.path.exists(extract_path):
            os.makedirs(extract_path)

        # Unzip the file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for member in zip_ref.namelist():
                member_path = os.path.join(extract_path, os.path.relpath(member, start=os.path.commonpath(zip_ref.namelist())))
                if member.endswith('/'):
                    os.makedirs(member_path, exist_ok=True)
                else:
                    os.makedirs(os.path.dirname(member_path), exist_ok=True)
                    with open(member_path, 'wb') as f:
                        f.write(zip_ref.read(member))

        # Clean up the zip file
        os.remove(zip_path)
        print("Download and extraction complete.")

def download(url, file_path):
    # Check if the file already exists
    if os.path.exists(file_path):
        print(f"File {file_path} already exists.")
    else:
        print("Downloading the file...")
        # Download the file
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte

        with open(file_path, 'wb') as file, tqdm(
            desc=file_path,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(block_size):
                file.write(data)
                bar.update(len(data))

        print("Download complete.")


# Download the Finnish Wikipedia dataset
url = "https://www.kielipankki.fi/download/wikipedia-fi/wikipedia-fi-2017-src/wikipedia-fi-2017-src.zip"
zip_path = "wikipedia-fi-2017-src.zip"
extract_path = "data/wikipedia-fi-2017-src"
download_and_extract(url, zip_path, extract_path)

"""
/data/wikipedia-fi-2017-src/wiki_partX.VRT
"""

################################################################################

# Download Finnish Stopwords
url = "http://members.unine.ch/jacques.savoy/clef/finnishST.txt"
download(url, "data/finnishST.txt")

################################################################################

# Download Finnish Stemmer
url = "http://members.unine.ch/jacques.savoy/clef/finnishStemmer.txt"
download(url, "data/finnishStemmer.txt")

################################################################################

# Download FinnWordNet
url = "https://www.kielipankki.fi/download/FinnWordNet/v2.0/FinnWordNet-2.0.zip"
zip_path = "FinnWordNet-2.0.zip"
extract_path = "data/FinnWordNet"
download_and_extract(url, zip_path, extract_path)

################################################################################

# Loading the FastText model
ftutil.download_model('fi', if_exists='ignore')  # Downloads the Finnish model
fin_model = ft.load_model('cc.fi.300.bin')
print(f"Model dimension: {fin_model.get_dimension()}")
ftutil.reduce_model(fin_model, 50)  # Reduce the model size to 100 dimensions
print(f"Model dimension after reduction: {fin_model.get_dimension()}")

################################################################################
################################################################################
################################################################################

# Task 1
print("Task 1: Search and Highlight\n")
keywords = ["ilmastonmuutos", "päästö", "joustavuus", "kestävyys"]

# def sah(keyword, data_path, parts_range):
#     linked_entity_pattern = re.compile(r'<a href="([^"]+)">([^<]+)</a>')  # Example pattern for linked entities
#     articles_with_links = []

#     # Loop through the dataset parts
#     for part_num in range(parts_range):
#         part_file = os.path.join(data_path, f"wiki_part{part_num}.VRT")

#         # Check if the part file exists
#         if not os.path.exists(part_file):
#             print(f"File does not exist: {part_file}")
#             continue

#         with open(part_file, 'r', encoding='utf-8') as file:
#             content = file.read()
#             # Find the article that matches the keyword
#             if keyword.lower() in content.lower():
#                 matches = linked_entity_pattern.findall(content)
#                 highlighted_content = content
#                 for link, entity in matches:
#                     highlighted_content = highlighted_content.replace(entity, f"**{entity}**")
#                 articles_with_links.append(highlighted_content)

#     return articles_with_links

# data_path = "data/wikipedia-fi-2017-src/"
# parts_range = 65
# for keyword in keywords:
#     print(f"Searching for keyword: {keyword}")
#     highlighted_articles = sah(keyword, data_path, parts_range)
#     for article in highlighted_articles:
#         print(article)
#         print("\n" + "#" * 80 + "\n")
"""
dataset_directory = 'data/wikipedia-fi-2017-src/'
# Function to read and search for a specific word in the dataset
def search_wikipedia_pages(word, dataset_directory):
    best_matching_page = None
    highest_similarity = -np.inf

    word_vector = fin_model.get_sentence_vector(word)

    # Iterate over all dataset parts
    for filename in os.listdir(dataset_directory):
        if filename.startswith("wiki_part") and filename.endswith(".VRT"):
            file_path = os.path.join(dataset_directory, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                pages = file.read().split('<page>')

                for page in pages:
                    title_match = re.search(r'<title>(.*?)</title>', page)
                    if title_match:
                        title = title_match.group(1)
                        page_vector = fin_model.get_sentence_vector(title)

                        similarity = np.dot(word_vector, page_vector)

                        print(f'Similarity between "{word}" and page title "{title}": {similarity}')

                        if similarity > highest_similarity:
                            highest_similarity = similarity
                            best_matching_page = page
    return best_matching_page

# Function to highlight linked entities
def highlight_linked_entities(page_content):
    if page_content is None:
        return "No matching page found."
    highlighted_content = re.sub(r'(<link entity=")(.*?)(">)(.*?)(</link>)', r'\1\2\3**\4**\5', page_content)
    return highlighted_content

# Output results
for word in keywords:
    best_page = search_wikipedia_pages(word, dataset_directory)
    if best_page is None:
        print(f'No matching page found for "{word}".\n')
    else:
        highlighted_page = highlight_linked_entities(best_page)
        print(f'Best matching page for "{word}":\n')
        print(highlighted_page)
    print("\n" + "-" * 80 + "\n")

"""

# Define the Finnish translations of the environmental terms
terms = {
    'ilmastonmuutos',
    'päästö',
    'joustavuus',
    'ekologinen kestävyys'
}


# Function to load the extracted Wikipedia dataset
def load_wikipedia_dataset(directory):
    data = {}
    data_titles = {}

    # Iterate over all dataset parts
    for filename in os.listdir(directory):
        if filename.startswith("wiki_part") and filename.endswith(".VRT"):
            file_path = os.path.join(directory, filename)
            print(f"Processing file: {file_path}")

            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()

                # Use regex to find all documents within the <doc>...</doc> tags
                docs = re.findall(r'<doc id="(.*?)" url="(.*?)" title="(.*?)">(.*?)</doc>', content, re.DOTALL)

                # Process each document found
                for doc_id, url, title, doc_content in docs:
                    data_titles[title.casefold()] = {doc_id, url}

                    # Only store the data if the title matches one of the specified terms
                    if title.casefold() in terms:
                        data[title.casefold()] = {
                            'id': doc_id,
                            'url': url,
                            'content': doc_content,
                            'file_path': file_path
                        }

    return data, data_titles

dataset_directory = "data/wikipedia-fi-2017-src"  # Path to the extracted dataset
data, data_titles = load_wikipedia_dataset(dataset_directory)

# print(f"Loaded data: {data.items()}")

#highlighted_content = re.sub(r'(<link entity=")(.*?)(">)(.*?)(</link>)', r'\1\2\3**\4**\5', data['ilmastonmuutos']['content'])
#print(f"Highlighted Content: {highlighted_content}")

# Function to highlight linked entities
def highlight_linked_entities(content):
    highlighted_content = re.sub(r'(<link entity=")(.*?)(">)(.*?)(</link>)', r'\1\2\3**\4**\5', content)

    return highlighted_content

for title, info in data.items():
    highlighted_content = highlight_linked_entities(info['content'])
    print(f"Title: {title}, highlighted Content: {highlighted_content}")


# Function to extract third column data from sentences
def extract_third_column(sentence):
    third_column_data = []
    lines = sentence.strip().split('\n')
    for line in lines:
        if line.strip() and not line.startswith('<'):
            columns = line.split('\t')
            if len(columns) > 2:
                third_column_data.append(columns[2])
    return third_column_data

def process_paragraphs(input_text):
    #print(f"Input Text: {input_text}")
    paragraphs = re.findall(r'<paragraph>(.*?)</paragraph>', input_text, re.DOTALL)
    all_third_column_data = []
    #print(f"Paragraphs: {paragraphs}")
    for paragraph in paragraphs:
        sentences = re.findall(r'<sentence>(.*?)</sentence>', paragraph, re.DOTALL)
        #print(f"Sentences: {sentences}")
        for sentence in sentences:
            #print(f"Sentence: {sentence}")
            third_column_data = extract_third_column(sentence)
            all_third_column_data.append(' '.join(third_column_data))
    return all_third_column_data

#print(f"Wikipedia data: {wikipedia_data['ilmastonmuutos']['content']}")

# Process each item in wikipedia_data

for item in data.items():
    third_column_data = process_paragraphs(item[1]['content'])
    print(f"Item {item} third column data:")
    print(third_column_data)
    print("\n" + "#" * 80 + "\n")


sentences = re.findall(r'<sentence>(.*?)</sentence>', data['ilmastonmuutos']['content'], re.DOTALL)
all_words = []

if sentences:
    for idx, sentence_content in enumerate(sentences):
        print(f"Processing sentence {idx + 1}...")

        # Split each sentence into individual lines
        lines = sentence_content.strip().split('\n')

        # Extract the word (3nd column) from each line
        words = [line.split('\t')[2] for line in lines if line.strip()]
        all_words.extend(words)
else:
    print("No sentences found in the document.")

print(f"Words: {all_words}")


################################################################################
################################################################################
################################################################################

# Task 2

# Preprocessing function to remove stopwords, stemming, and tokenize the document
def preProcess(doc, stopwords):
    stopwords_set = set(stopwords)
    sentences = sent_tokenize(doc.casefold())

    tokens = []
    for sentence in sentences:
        words = word_tokenize(sentence)
        # Add stemmer here
        # words = [stemmer.stem(word) for word in words]

        words = [word for word in words if word.isalpha() and word not in stopwords_set]
        tokens.extend(words)

    return ' '.join(tokens)

# Function to read the stopwords from the file
def load_stopwords(file_path):
    with open(file_path, 'r', encoding='ISO-8859-1') as f:
        stopwords = f.read().splitlines()
        print(f"Loaded {len(stopwords)} stopwords.")
        print(f"First 10 stopwords: {stopwords[:10]}")
    return stopwords

# Load Finnish stopwords
stopwords_path = "data/finnishST.txt"
stopwords = load_stopwords(stopwords_path)

# Document download
# TO-DO

documents = [
    "Tämä on esimerkki dokumentista. Dokumentti sisältää tekstiä, jota käytetään dokumenttien esimerkkinä.",
    "Tämä on toinen esimerkki dokumentista. Dokumentti sisältää myös tekstiä, jota käytetään dokumenttien esimerkkinä."
]

# Preprocess the document
processed_docs = [preProcess(doc, stopwords) for doc in documents]

# Print the processed document
for i, doc in enumerate(processed_docs):
    print(f"Processed Document {i+1}:")
    print(doc)
    print("\n")

# TF-IDF Vectorization of each document
tf = TfidfVectorizer(use_idf=True, min_df=1)
tfidf_matrix = tf.fit_transform(processed_docs)

# Compute cosine similarity of each pair of these documents
cosine_sim = cosine_similarity(tfidf_matrix)

# Provide the result in a table and comment on the findings.


################################################################################
################################################################################
################################################################################

# Task 3

# Preprocessing function to remove stopwords, stemming, and tokenize the document
# From task2

def extract_clickable_entities(doc):
    # TO-DO
    return

# Preprocessing function to extract clickable entities and process them
def preProcess_entities(doc):
    entities = extract_clickable_entities(doc)
    return preProcess(entities, stopwords)

# Function to read the stopwords from the file
    # From task2

# Load Finnish stopwords
stopwords_path = "data/finnishST.txt"
stopwords = load_stopwords(stopwords_path)

# Document download
# TO-DO

documents = [
    "Tämä on esimerkki dokumentista. Dokumentti sisältää tekstiä, jota käytetään dokumenttien esimerkkinä.",
    "Tämä on toinen esimerkki dokumentista. Dokumentti sisältää myös tekstiä, jota käytetään dokumenttien esimerkkinä."
]


# Load and preprocess each document, restricting to clickable entities
processed_entity_docs = [preProcess_entities(doc) for doc in documents]

# TF-IDF Vectorization of each document
tf = TfidfVectorizer(use_idf=True, min_df=1)
tfidf_matrix = tf.fit_transform(processed_docs)

# Compute cosine similarity of each pair of these documents
cosine_sim = cosine_similarity(tfidf_matrix)

# Provide the result in a table and comment on the findings.


################################################################################
################################################################################
################################################################################

# Task 4

