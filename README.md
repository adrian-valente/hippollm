# HippoLLM: a hybrid graph-vector database for LLM memory

This is a prototype of (still experimental) project of a hybrid database system that can serve as a robust memory of facts for LLMs that can easily be inspected by humans. This projects starts from the premise that current methods to use LLMs on proprietary sets of unstructured text, which are mainly RAG and fine-tuning, fall short of providing a way for the LLM to synthesize the information found in these texts in a format that allows easy retrieval both for the LLM and for humans. The solution we propose here is to build a hybrid graph-vector database, structured both as a knowledge graph, with entities and relations between entities (here, any natural language sentence can be a relation), and as a vector store, in which relations can be retrieved from their embeddings with fast vector similarity search (powered by ChromaDB).

![Figure 1](assets/hippofig.pdf)

More information can be found in the abstract and poster available in `assets/`

## Usage
TBD.

## Citation
For the moment, you can cite:

> Valente, A., "hippoLLM: Scrutable and Robust Memory for LLMs with Hybrid Graph-Vector Databases", HyCHA'24, Gif-sur-Yvette, France. https://hycha24.sciencesconf.org/ 

## Resources
The following amazing resources have been used:
- [langchain](https://github.com/langchain-ai/langchain)
- [ChromaDB](https://github.com/chroma-core/chroma)
- [sentence-transformers](https://github.com/UKPLab/sentence-transformers)
- [Mistral 7B](https://huggingface.co/mistralai/Mistral-7B-v0.1)
- [ollama](https://github.com/ollama/ollama)
