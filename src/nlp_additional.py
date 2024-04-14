from sentence_transformers import CrossEncoder
from typing import Optional

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
        
    def top_entailment(self, item: str, others: list[str]) -> Optional[int]:
        """Return the index of the element in others that most likely entails item."""
        scores = self.entailment_model.predict([(other, item) for other in others])
        do_entail = [scores[i].argmax() == 1 for i in range(len(others))]
        
        
        if any(do_entail):
            scores_entail = [scores[i][1] for i in range(len(scores))]
            # get the argmax
            argmx = -1
            max_score = -float('inf')
            for i in range(len(scores_entail)):
                if scores_entail[i] > max_score and do_entail[i]:
                    argmx = i
                    max_score = scores_entail[i]
            return argmx
        else:
            return None
        
    