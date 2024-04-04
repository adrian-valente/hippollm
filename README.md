# HippoLLM: a hybrid graph-vector database for LLM memory

This is a prototype of (still experimental) project of a hybrid database system that can serve as a robust memory of facts for LLMs that can easily be inspected by humans. This projects starts from the premise that current methods to use LLMs on proprietary sets of unstructured text, which are mainly RAG and fine-tuning, fall short of providing a way for the LLM to synthesize the information found in these texts in a format that allows easy retrieval both for the LLM and for humans. The solution we propose here is to build a hybrid graph-vector database, structured both as a knowledge graph, with entities and relations between entities (here, any natural language sentence can be a relation), and as a vector store, in which relations can be retrieved from their embeddings with fast vector similarity search (powered by ChromaDB).

![Figure 1](assets/hippofig.jpg?raw=True)

More information can be found in the [abstract](assets/abstract_hycha24_1.pdf) and [poster](assets/poster.pdf).

## Installation
To install the program, clone the repository and create the corresponding conda environment:

```sh
conda env create -f environment.yml
```

The system uses ollama to run the LLM efficiently, and in this case Mistral 7B is used, so ollama should download mistral weights and then be launched in the background, for example with:
```sh
ollama pull mistral
ollama serve&
```
(where you then have to use the process id if you want to stop the Ollama server). That should be sufficient so that invocations of mistral from the code work! 

## Usage
This system works in two steps: in annotation mode, facts are extracted from unstructured text data and embedded in the hybrid memory system. In retrieval mode, a question can be asked, the most relevant facts in the memory will be retrieved and displayed, and an answer can then be generated using these facts as a source of knowledge.

So far the prototype runs as described in the poster for fact extraction, relying only on the zero-shot capacities of Mistral 7B and an additional NLI model (NLI-DeBERTa v3).

You can for example try to annotate a wikipedia page, and then ask questions about the subject:
```sh
python annotate_wikipedia.py Paris wikidb
python retrieval.py  # User will then be prompted for questions
```

The `annotate_wikipedia.py` script takes two arguments, a query for an article (first matching result will then be parsed and annotated) and the location of the database.

You can also add your own text files (for example in .txt format). Note that you can annotate as many documents as you want in a single database.

Additional scripts to interact with a database directly (notably visually) will soon be added.

## Experiments
For the moment I have experimented with adding titles and abstracts from wikipedia-en as entities, in particular a subset of approximately 1.5M most visited articles (selected as those mentioned in a pageviews dump of a single day). The database with those entities added can be created with the script bootstrap_wikipedia.py, for example:
```
python boostrap_wikipedia.py --db_loc wikidb --dump_loc tmp
```

So far, the abstracts obtained through the official dump are however often inconsistent or uninformative, and I am still looking for a better source of the good quality abstracts to improve entity linking.

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
