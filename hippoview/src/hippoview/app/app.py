import collections
import itertools
import json
import os
from typing import cast, Literal, Optional
from typing_extensions import get_args
from functools import lru_cache

import pandas as pd
from flask import Flask
from flask_cors import CORS, cross_origin

from langchain_core.embeddings import Embeddings
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings

from hippollm.storage import EntityStore

__all__ = ["Search", "create_app"]

TQueryType = Literal["entity", "fact"]

class Search:
    """Search over KG."""

    def __init__(self, path: os.PathLike, emb: str) -> None:

        self.colors = ["#00A36C", "#9370DB", "#bbae98", "#7393B3", "#677179", "#318ce7", "#088F8F"]
        
        embed_model = SentenceTransformerEmbeddings(model_name=emb)
        self.storage = EntityStore(embedding_model=embed_model, persist_dir=path)
        print(f"Loaded db with {len(self.storage.entities)} entities and {len(self.storage.facts)} facts.")
        self.metadata = {}

        self.triples = collections.defaultdict(tuple)
        self.relations = collections.defaultdict(list)

    def explore(self, 
                origin: str, 
                relations: list[tuple[str, list[int], str]], 
                visited: set[str],
                depth: int, 
                max_depth: int,
                max_relations: Optional[int] = None,
                ) -> list[str]:
        depth += 1
        neighbour_facts = self.storage.get_neighbours(origin, return_facts=True)
        if max_relations is not None:
            neighbour_facts = sorted(
                neighbour_facts, key=lambda x: len(x[1]), reverse=True
                )[:max_relations]
        
        for neighbour, facts in neighbour_facts:
            relations += [tuple([origin, facts, neighbour])]
            if depth < max_depth and neighbour not in visited:
                visited.add(neighbour)
                relations = self.explore(
                    neighbour,
                    relations,
                    visited,
                    depth,
                    max_depth,
                )
        return relations

    def __call__(self, 
                 query: str, 
                 query_type: TQueryType, 
                 k: int, 
                 n: int, 
                 p: Optional[int] = None,
                 ) -> dict[str, list[dict[str, str]]]:
        nodes, links = [], []
        nodes_map = {}
        # prune = collections.defaultdict(int)

        entities = []
        facts = []
        for q in query.split(";"):
            q = q.strip()
            if query_type == "entity":
                ents = self.storage.get_closest_entities(q, k)
                print(ents)
                entities.extend(ents)
            elif query_type == "fact":
                fcts = self.storage.get_closest_facts(q, k)
                ents = set()
                for fact in fcts:
                    ents.update(fact.entities)
                entities.extend([self.storage.get_entity(e) for e in ents])
                facts.extend(fcts)
                

        # Add nodes
        for group, e in enumerate(entities):
            node = {
                    "id": e.name,
                    "group": group,
                    "color": "#960018",
                    "fontWeight": "bold",
                    "description": e.description,
                    "facts": {},
                }
            nodes.append(node)
            nodes_map[e.name] = node
            

        # Search for neighbours (in entity query mode only)
        if query_type == "entity":
            already_seen = {e.name for e in entities}
            added_to_plot = already_seen.copy()
            for group, e in enumerate(entities):
                color = self.colors[group % len(self.colors)]
                relations = self.explore(e.name, [], already_seen, 0, n, p)
                for a, fcts, b in list(relations):
                    for x in (a, b):
                        if x not in added_to_plot:
                            x_ent = self.storage.get_entity(x)
                            node = {
                                "id": x,
                                "group": group,
                                "color": color,
                                "description": x_ent.description,
                                "facts": {},
                            }
                            nodes.append(node)
                            nodes_map[x] = node
                            added_to_plot.add(x)
                    links.append(
                        {
                            "source": a,
                            "target": b,
                            "value": len(fcts),
                            "relation": str(fcts),
                            "facts": {
                                str(fc): self.storage.get_fact(fc).text for fc in fcts
                            },
                        }
                    )
                    for fc in fcts:
                        nodes_map[a]["facts"][str(fc)] = self.storage.get_fact(fc).text
                        nodes_map[b]["facts"][str(fc)] = self.storage.get_fact(fc).text
        
        # Add links (in fact query mode only)
        if query_type == "fact":
            links_map = {}
            for fact in facts:
                # Add all links between entities
                for (a, b) in itertools.combinations(fact.entities, 2):
                    if frozenset([a, b]) not in links_map:
                        link = {
                            "source": a,
                            "target": b,
                            "value": 1,
                            "relation": str(fact.id),
                            "facts": {str(fact.id): fact.text},
                        }
                        links.append(link)
                        links_map[frozenset([a, b])] = link
                    else:
                        links_map[frozenset([a, b])]["value"] += 1
                        links_map[frozenset([a, b])]["facts"][str(fact.id)] = fact.text
                 
                # Add fact to nodes    
                for e in fact.entities:
                    nodes_map[e]["facts"][str(fact.id)] = fact.text

        return {"nodes": nodes, "links": links}


def create_app(path: os.PathLike, emb: str) -> Flask:
    app = Flask(__name__)
    app.config["JSONIFY_PRETTYPRINT_REGULAR"] = True
    app.config["CORS_HEADERS"] = "Content-Type"
    CORS(app, resources={r"/search/*": {"origins": "*"}})
    
    search = Search(path, emb)

    @app.route("/search/<k>/<n>/<p>/<query_type>/<query>", methods=["GET"])
    @cross_origin()
    def get(k: int, n: int, p: int, query: str, query_type: str):
        assert query_type in get_args(TQueryType)
        query_type = cast(TQueryType, query_type)
        return json.dumps(search(query=query, query_type=query_type, k=int(k), n=int(n), p=int(p)))

    return app
