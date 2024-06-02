# Hippoview: graphical interface for hippoLLM

*Credits: The code is entirely adapted from the repository: [kgsearch](https://github.com/raphaelsty/kgsearch) by [@raphaelsty](https://github.com/raphaelsty). The compelling visualization is done thanks to the Javascript [3D Force Graph library](https://github.com/vasturiano/3d-force-graph)*

The hippoview tool directly extends the hippoLLM project with a graphical interface, dedicated to visually explore hybrid graph-vector databases (GVDB). You should have installed hippoLLM first, and have a built GVDB somewhere on your disk (one is provided in the examples of the hippollm project).

![](kgsearch.gif)

KGSearch is a minimalist tool for searching and viewing entities in a graph and is dedicated to a local environment. The application provides a Python client with three distinct terminal commands: `add, start, open`. The application default proposes to search through the knowledge graph [Countries](https://www.aaai.org/ocs/index.php/SSS/SSS15/paper/view/10257/10026). You can explore the borders we must cross to get from one country to another and see how small the 🌍 is.

## Installation

The main hippollm project must already be installed. If that is the case, with the corresponding virtual environment activated, go to the hippoview directory and run:

```sh
pip install -e .
```

## Usage

Once installed you can open the interface by running:

```sh
python -m hippoview start --path path_to_db [--emb embedding_model]
```
where `path_to_db` is the path to the directory of a GVDB, and `embedding_model` the name of a SentenceTransformers model (if not present in the `metadata.yaml` of the DB).

The interface then works with two types of queries: entity queries, and fact queries. In the first textbox, you can ask for one or several entities (then separated by a ;. Example: "Paris; Eiffel tower"), and the closest entities found to your query will be shown. In the second textbox, you can put any general sentence or question, that will trigger a search for the closest related facts (example: "Where does the name Paris come from?"). Those facts, and the related entities, will then be plotted (note that the other facts and neighbors of those entities not related to the question will not be plotted). Only one of these two types of queries can be done at a time.

Additional parameters :
- Top K: the number of answers entities/facts retrieved for any given query (set to somewhere between 5 and 10 for facts).
- Steps: number of multi-hops to plot, ie. neighbors of neighbors.
- Max Neighbors: max number of relations to plot per node, for nodes with lots of neighbors.

When hovering over a node, the facts associated to this node and to the query will be shown. When hovering over a link, the facts associated to both nodes and retrieved in the query will be shown. Note that our databases are hypergraphs, so a fact can be associated to many entities! In the visualization, an edge simply signifies that at least one fact involves those two entities.

