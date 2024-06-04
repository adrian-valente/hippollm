# HippoLLM: a hybrid graph-vector database for LLM memory

This is a prototype of (still experimental) project of a hybrid database system that can serve as a robust memory of facts for LLMs that can easily be inspected by humans. This projects starts from the premise that current methods to use LLMs on proprietary sets of unstructured text, which are mainly RAG and fine-tuning, fall short of providing a way for the LLM to synthesize the information found in these texts in a format that allows easy retrieval both for the LLM and for humans. The solution we propose here is to build a hybrid graph-vector database, structured both as a knowledge graph, with entities and relations between entities (here, any natural language sentence can be a relation), and as a vector store, in which relations can be retrieved from their embeddings with fast vector similarity search (powered by ChromaDB).

![Figure 1](assets/hippofig.jpg?raw=True)

More information can be found in the [abstract](assets/abstract_hycha24_1.pdf) and [poster](assets/poster.pdf).

## Installation
To install the program, clone the repository and create a conda environment for this project (for example with `venv` or `conda`), and then do a local installation with `pip`. At least one backend should be installed, and several backends can be installed if specified between the brackets (see [available backends](#backends)). For example:

```sh
conda create -n hippo_env python=3.11
conda activate hippo_env
pip install -e '.[llama-cpp, openai]'
python install.sh  # Install punkt
```

You can then also install our graphical front-end extension [hippoview](#frontend).

## Usage
This system works in two steps: in annotation mode, facts are extracted from unstructured text data and embedded in the hybrid memory system. In retrieval mode, a question can be asked, the most relevant facts in the memory will be retrieved and displayed, and an answer can then be generated using these facts as a source of knowledge.

The program requires an LLM to run in annotation mode, which can be run locally or from an online provider (OpenAI, Groq...). Internally, the program also uses an embedding model from SentenceTransformers (by default all-MiniLM-L6-v2) and an additional NLI model (NLI-DeBERTa v3).

You can for example try to annotate a wikipedia page, and then ask questions about the subject:
```sh
python hippollm.annotate_wikipedia Rome /path/to/stored_db \
  --llm_backend openai --llm_model gpt-3.5-turbo
```

Note that you can also use one of our configurations (and use them as examples of yaml configurations). For example to run local inference with a Mistral model and llama-cpp, if you are in the root of this repository:
```sh
python hippollm.annotate_wikipedia Rome /path/to/stored_db \
  --cfg configs/mistral.yaml --llm_model /path/to/mistral_weights.gguf
```

To start the retrieval mode, simply provide a previously built database. One is provided in the examples of this repository, running by default with the OpenAI backend (you would need an API key in your environment variables):
```sh
python hippollm.retrieval examples/wikipedia_Paris
```

But you can supersede the default arguments:
```sh
python hippollm.retrieval examples/wikipedia_Paris \
  --llm_backend llama_cpp --llm_model /path/to/mistral_weights.gguf
```


## Backends
### OpenAI/Groq
The OpenAI backend is the easiest way to try the system if you have some credits on their API (the Groq backend is also already available, and we are just waiting for updates on their API). To use, simply make sure you have your `OPENAI_API_KEY` in your environment, and then run a query, for example:

```sh
python -m hippollm.annotate_wikipedia Paris path/to/db \
  --llm_backend openai --llm_model gpt-3.5-turbo
```

### Llama-cpp
The llama-cpp is the best way to run powerful LLMs locally for the moment, and the only backend to support grammar-guided generation. To run with for example Mistral 7B, you should download the GGUF weights, for example [here](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF), and then run the annotator with the path pointing to those files, for example:

```sh
python3 -m hippollm.annotate_txt file.txt path/to/db \
  --llm_backend llama_cpp --llm_model path/to/mistral-7b-instruct-v0.2.Q4_0.gguf
```

### Ollama
The Ollama backend is another possibility to run powerful LLMs locally and without much hassle. You can try with for example Mistral 7B, so ollama should download mistral weights and then be launched in the background, for example with:
```sh
ollama pull mistral
ollama serve&
```
(where you then have to use the process id if you want to stop the Ollama server). That should be sufficient so that invocations of mistral from the code work! 

## Frontend

We provide a front-end extension in this repository that is installable separately, see the [corresponding README](hippoview/README.md). Its code has been directly adapted from the [kgsearch repository](https://github.com/raphaelsty/kgsearch) by [@raphaelsty](https://github.com/raphaelsty) and is built with React and Flask.

## Experiments

For the moment I have experimented with adding titles and abstracts from wikipedia-en as entities, in particular a subset of approximately 1.5M most visited articles (selected as those mentioned in a pageviews dump of a single day). The database with those entities added can be created with the script bootstrap_wikipedia.py, and it should take about 6 hours to embed all of them with a consumer GPU, with the command for example:
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
- [Mistral 7B](https://huggingface.co/mistralai/Mistral-7B-v0.1) was used a lot during development
- [llama.cpp](https://github.com/ggerganov/llama.cpp) and the bindings [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
- [ollama](https://github.com/ollama/ollama) 
- [kgsearch](https://github.com/raphaelsty/kgsearch) for the visual interface
