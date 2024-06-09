
## Introduction
This repository contains code for various natural language processing tasks, including text generation, summarization, question answering, and context retrieval. The code utilizes pre-trained models from Hugging Face Transformers and Google's Sentence Transformers, along with ChromaDB for context retrieval.

## Installation
Ensure you have Python installed, along with the necessary packages. You can install the required packages via pip:

```bash
!pip install -q chromadb sentence-transformers pymupdf
!pip install -q langchain
```

## Usage
### Setting up GPU
Check if GPU is available:

```python
print(torch.cuda.is_available())
```

### Hugging Face Login
```python
access_token = "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"
login(access_token)
```

### Available Models
- GPT2
- T5
- BERT
- DistilBERT
- GPT-Neo
- Gemma

### Usage Example
Instantiate a model:

```python
llm = GPT2()
```

Generate text:

```python
response = llm.generate_text(question, context)
print(response)
```

### Collection
The `Collection` class facilitates context retrieval using ChromaDB. Here's how to use it:

```python
collection = Collection('rag')
collection.add_contexts(context_data)
response = collection.retrieve_contexts('amazon', top_n=2)
print(response)
```

### RAG (Retrieval-Augmented Generation)
The `RAG` class combines an LLM with a Collection for generating responses based on given queries. Here's an example:

```python
rag = RAG(llm, collection)
response = rag.generate_response(query, top_n=3)
print(response)
```

### Summarizer
The `Summarizer` class provides methods for text summarization using different models (T5, BART, Pegasus). Here's how to use it:

```python
summarizer = T5_Summarizer(load_online=True)
summary = summarizer.summarize_text(context)
print(summary)
```

### PDF Processing
The `extract_text_from_pdf` function extracts text from a PDF file, and `preprocess_text` function splits the text into manageable chunks.

```python
pdf_path = "path_to_pdf_file.pdf"
text = extract_text_from_pdf(pdf_path)
contexts = preprocess_text(text)
```

# Function Documentation

## 1. `extract_text_from_pdf(pdf_path)`

**Description:**
Extracts text content from a PDF file.

**Parameters:**
- `pdf_path` (str): Path to the PDF file.

**Returns:**
- `text` (str): Extracted text content from the PDF.

## 2. `preprocess_text(text, chunk_size=500)`

**Description:**
Preprocesses text content, splitting it into manageable chunks.

**Parameters:**
- `text` (str): The text to be processed.
- `chunk_size` (int): Size of each chunk (default is 500 words).

**Returns:**
- `preprocessed_text_chunks` (list): List of preprocessed text chunks.

## 3. `Collection` class

### `__init__(self, collection_name: str, transformer_type: str = 'all-MiniLM-L6-v2', load_online=False, save_transformer=False)`

**Description:**
Initializes a collection object.

**Parameters:**
- `collection_name` (str): Name of the collection.
- `transformer_type` (str): Type of sentence transformer to use (default is 'all-MiniLM-L6-v2').
- `load_online` (bool): Whether to load the transformer online.
- `save_transformer` (bool): Whether to save the transformer locally after loading.

### `load_sentence_transformer(self, transformer_type: str, load_online: bool, save_transformer: bool)`

**Description:**
Loads the sentence transformer.

**Parameters:**
- `transformer_type` (str): Type of sentence transformer to load.
- `load_online` (bool): Whether to load the transformer online.
- `save_transformer` (bool): Whether to save the transformer locally after loading.

### `add_contexts(self, context_data: list)`

**Description:**
Adds context data to the collection.

**Parameters:**
- `context_data` (list): List of text contexts to be added.

### `retrieve_contexts(self, question: str, top_n: int = 1)`

**Description:**
Retrieves relevant contexts from the collection based on a given question.

**Parameters:**
- `question` (str): The question for which contexts are retrieved.
- `top_n` (int): Number of top contexts to retrieve (default is 1).

## 4. `Summarizer` class

### `__init__(self, summarizer_model:str='t5', load_online=False, save_model=False)`

**Description:**
Initializes a summarizer object.

**Parameters:**
- `summarizer_model` (str): Type of summarizer model to use (default is 't5').
- `load_online` (bool): Whether to load the model online.
- `save_model` (bool): Whether to save the model locally after loading.

### `load_summarizer(self, summarizer_model: str, load_online: bool, save_model: bool)`

**Description:**
Loads the summarizer model.

**Parameters:**
- `summarizer_model` (str): Type of summarizer model to load.
- `load_online` (bool): Whether to load the model online.
- `save_model` (bool): Whether to save the model locally after loading.

### `summarize_text(self, input_text: str, context: str = '')`

**Description:**
Summarizes the input text using the loaded summarizer model.

**Parameters:**
- `input_text` (str): The text to be summarized.
- `context` (str): Additional context for summarization (optional).

### `free_memory(self)`

**Description:**
Frees up memory by deleting the loaded model and tokenizer, and clearing GPU memory.

## 5. `T5_Summarizer` class

This is a subclass of `Summarizer` specifically for T5 summarization. It inherits all methods from the `Summarizer` class and overrides the `summarize_text` method with T5-specific summarization logic.

## 6. Standalone Summarization Functions

### `summarize_with_bart(text: str)`

Summarizes text using the BART model.

**Parameters:**
- `text` (str): The text to be summarized.

### `summarize_with_t5(text: str)`

Summarizes text using the T5 model.

**Parameters:**
- `text` (str): The text to be summarized.

### `summarize_with_pegasus(text: str)`

Summarizes text using the Pegasus model.

**Parameters:**
- `text` (str): The text to be summarized.


## References
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [Sentence Transformers](https://github.com/UKPLab/sentence-transformers)
- [ChromaDB](https://github.com/google-research-datasets/chromadb)

```

This README provides an overview of the code, including installation instructions, usage examples, and references to relevant resources. Adjust the paths and configurations as needed to suit your environment and requirements. Let me know if you need further assistance!
