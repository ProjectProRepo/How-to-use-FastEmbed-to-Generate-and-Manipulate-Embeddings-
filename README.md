# Fastembed Tutorial: Generate & Use Text Embeddings with Qdrant, LangChain & More

This tutorial demonstrates how to generate and manipulate text embeddings using the `fastembed` Python library. It includes embedding generation, similarity computation, visualization, and integrations with tools like Qdrant, LangChain, and LlamaIndex.

## Overview

Fastembed is a high-performance embedding library built in Rust with Python bindings. It provides pre-trained embedding models that can be run on both CPU and GPU with ONNX under the hood. Fastembed integrates seamlessly with Qdrant, LangChain, and LlamaIndex, making it a practical tool for vector search, RAG systems, and AI pipelines.

## What You'll Learn

- How to install and use `fastembed` in Python
- How to generate embeddings from raw text
- How to visualize embeddings using PCA
- How to compute cosine similarity between embeddings
- How to perform semantic search using Qdrant
- How to use Fastembed with LangChain for retrieval
- How to use Fastembed with LlamaIndex for embedding documents

## Sections

### 1. Installing Fastembed

Instructions for installing `fastembed` with `pip`, with optional GPU support using the `[gpu]` extra. This ensures the embedding models can run efficiently in your environment.

### 2. Generating Embeddings

Use the `DefaultEmbedding` class from `fastembed.embedding` to load a pre-trained model and convert a list of text strings into embeddings. These embeddings are numerical vectors that capture semantic meaning.

### 3. Visualizing Embeddings

To better understand the relationships between generated embeddings, reduce their dimensions using PCA and visualize them in 2D using Matplotlib. This helps identify clusters and distances between text samples.

### 4. Comparing Embeddings

Use cosine similarity to measure how similar different text inputs are. A similarity matrix is computed using `sklearn.metrics.pairwise.cosine_similarity` and presented in a tabular format using Pandas.

### 5. Comparing a New Sentence

Generate an embedding for a new piece of text and compare it to existing embeddings to determine which existing input is most semantically similar to the new sentence.

### 6. Fastembed with Qdrant

Connect `fastembed` to Qdrant, a high-performance vector database. Create a collection, upload embeddings, and perform similarity search queries using vector distance. This is useful for building retrieval-based applications.

### 7. Fastembed with LangChain

Use `FastEmbedEmbeddings` from LangChain to embed text queries and documents. This enables integration with LangChain's retrieval components and agent architectures that rely on semantic understanding.

### 8. Fastembed with LlamaIndex

Integrate Fastembed with LlamaIndex to embed documents and perform efficient retrieval. This allows for lightweight, fast, and production-ready implementations of retrieval-augmented generation (RAG) workflows.

## Why Use Fastembed

- Minimal setup and fast performance
- Rust-backed with ONNX for cross-platform efficiency
- Works on CPU and GPU
- Supports multiple popular sentence embedding models
- Compatible with Qdrant, LangChain, and LlamaIndex

## Additional Resources

- GitHub: https://github.com/qdrant/fastembed  
- PyPI: https://pypi.org/project/fastembed/  
- Qdrant Docs: https://qdrant.tech/documentation/  
- LangChain Docs: https://python.langchain.com/  
- LlamaIndex Docs: https://docs.llamaindex.ai  

