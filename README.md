# NLP-Course-Project

## Group information

| Name             | Student ID | Email                      |
| ---------------- | ---------- | -------------------------- |
| Janne Yrjänäinen | Y58554010  | jyrjanai20@student.oulu.fi |
| Joona Syrjäkoski | Y58172266  | jsyrjako20@student.oulu.fi |
| Joonas Ojanen    | 2305882    | jojanen20@student.oulu.fi  |
| Lasse Rapo       | Y58553703  | lrapo20@student.oulu.fi    |

## Project description

Project 23: Environment keyword mapping

This project aims to review selected environmental topics according to Finnish Wikipedia-based description,
available in https://www.kielipankki.fi/download/wikipedia-fi/wikipedia-fi-2017-src/, containing all Finnish
Wikipedia articles available in January 2018. The dataset is provided with a set of linked entities, where each
entity links to a given page.

1. Consider the wordings: “climate change”, “emission”, “resilience”, “sustainability” (need to consider
their Finnish translations). Suggest a script where your input each of these wordings and output the
corresponding Wikipedia pages, highlighting all the (linked) entities in these pages. You may look at the
examples shown in [Python for NLP: Working with Facebook FastText Library]([https://stackabuse.com/python-for-nlp-working-with-facebook-fasttext-library/]).

2. Assume the content of each webpage is a single document. Use relevant NLTK script to create a corpus constituted of the four document, and appropriate proprocessing and lemmatization, to construct the TfIdfVectorizer of each document and then calculate the cosine similarity of each pair of these documents. Provide the result in a table and comment on the findings.

3. Repeat 2) when the documents are restricted only to clickable entities of each document.

4. Write a script that explores the clickable entities in each of the four concepts (exploring only once), and generates new extended vocabulary for each concepts, and compute the new similarity between each pair of concepts using extent of overlapping of overall vocabulary and reduced vocabulary (when restricting the vocabulary to clickable entities). We shall refer to the case where the clickable entities are further explored as extended vocabulary case.

5. We want to assess the importance of each of the four concepts by the number of clickable entities including in the webpage of the given concept. Write a script that implements this strategy to determine the importance of each concept. Next, we want to use the extended vocabulary by quantifying the importance of individual entity in the original webpage by the number of clickable links it generates. Summarize a table highlighting the order of importance of the four concepts according to each of the original and extended vocabulary.

6. We want to assess the similarity between the concepts is reflected in the their webpage content. Use a script to calculate Wu and Palmer WordNet semantic similarity between each pair of the four concepts and then compare result with the Jaccard similarity obtained by both original vocabulary in 2) and extended vocabulary 4), and comment on the compatibility between the semantic similarity and the above constructed Jaccard similarity measures.

7. Now we want to evaluate the emotion of the text in each dataframe. For this purpose, use WordnetAffect https://github.com/clemtoy/WNAffect), which generates an emotion state and the part of speech tag of each word matched in the lexicon, and then compile the overall output for each concept. Repeat this process for both restricted and extended vocabulary cases, and comment on the compatibility between the emotion gained from the corresponding Wikipedia page and the intuitive emotion from individual concept.

8. We want to repeat this process for sentiment polarity. For this purpose, use the Vader sentiment https://github.com/cjhutto/vaderSentiment to assess the sentiment of each the Wikipedia page associated to each concept (both with original vocabulary and extended vocabulary, by aggregating the sentiment of individual
clickable entity’s page) to compute the sentiment associated with each concept. Comment on the sentiment of each concept in restricted and extended vocabulary and its compatibility with the intuitive sentiment gained from the inherent definition of these concepts.

9. We want to investigate the similarity using the embedding representation. For this purpose, write a program that uses word2vec embedding to evaluate i) the consistency of the Wikipedia page by
calculating the cosine similarity between the underlined concept and the average word2vec embedding of every words contained in the page, and ii) the similarity between the various concepts by computing
the cosine similarity between the aggregated embedding vector of the corresponding pages. Comment on the compatibility between webpage content and concept and similarity between the various concepts.

10. Repeat 9) when the extended vocabulary is considered.

11. Use appropriate literature to comment on the findings. Also, identify any additional input that would allow you to further elucidate any of the preceding, and use appropriate literature of corpus linguistic literature to justify your findings and comment on the obtained results. Finally, comment on the limitations and structural weakness of the data processing pipeline.


## Virtual environment

You can use either virtual environment or conda environment. Choose one of them. If you are translating the words from Finnish to English, and you would like to use GPU to speed up the translations, you should use conda environment. Otherwise, you can use virtual environment.

### Create virtual environment

```bash
python3 -m venv venv
```

### Activate virtual environment

Linux / macOS
```bash
source venv/bin/activate
```
or on Windows (CMD)
```bash
venv\Scripts\activate.bat
```
or on Windows (PowerShell)
```bash
venv\Scripts\Activate.ps1
```

### Deactivate virtual environment

```bash
deactivate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

## Conda environment

Install miniconda from [here](https://docs.conda.io/en/latest/miniconda.html)

### Open Conda terminal on Cmd

When installed to everyone
```bash
%ProgramData%\miniconda3\condabin\activate
```

or

When installed to user
```bash
%UserProfile%\miniconda3\condabin\activate
```

### Create conda environment

```bash
conda env create -f environment_cuda.yml
```

### Activate conda environment

```bash
conda activate cuda
```

### Deactivate conda environment

```bash
conda deactivate
```

## Development

### Formatting

Using Black formatter

```bash
black .
```




