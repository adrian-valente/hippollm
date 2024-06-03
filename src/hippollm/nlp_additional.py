from sentence_transformers import CrossEncoder


class NLPModels:
    """
    Wrapper for small NLP models used in the annotation process (to keep them in memory).
    So far only an entailment model (NLI) is wrapped.
    """
    
    def __init__(
        self,
        entailment_model: str = 'cross-encoder/nli-deberta-v3-base',
        ) -> None:
        self.entailment_model = CrossEncoder(entailment_model)
        
    def detect_entailment(self, item: str, other: str) -> bool:
        """Decide if there is item entails other."""
        scores = self.entailment_model.predict([(item, other)])
        if scores.argmax() == 1:
            return True
        else:
            return False
        
    def entailment_classify(self, item: str, others: list[str]) -> tuple[list[int], list[list[float]]]:
        """Return the index of the element in others that most likely entails item."""
        scores = self.entailment_model.predict([(other, item) for other in others])
        do_entail = [scores[i].argmax() == 1 for i in range(len(others))]
        scores_entail = [scores[i][1] for i in range(len(scores))]
        # argsort
        argsort_entail = sorted(range(len(scores_entail)), 
                                key=scores_entail.__getitem__,
                                reverse=True)
        sorted_items = [i for i in argsort_entail if do_entail[i]]
        return sorted_items, scores.tolist()
