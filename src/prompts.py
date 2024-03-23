from langchain.prompts import PromptTemplate
from helpers import itemize_list


contextualization_prompt = PromptTemplate.from_template(
    """Here is the beginning of a text. What is the type and subject of this text?
    {text}
    
    The text is """
)

annotation_prompt = PromptTemplate.from_template(
    """Here is an extract of a text of the following type: {context}
    Write simple statements summarizing all the facts stated in the following text. Make each statement as atomic as possible, and keeping the sentences as concise as possible. Extract only those facts that contain general knowledge, and not specificities of this text. Make sure each fact can be understood by itself without further context. Here is the text:
    {text}
    
    Facts:
    - """
)

confrontation_prompt = PromptTemplate.from_template(
    """Here are two facts:
    A) {fact}
    B) {other_fact}
    
    Classify those facts between the following categories: Equivalent, A generalizes B, B generalizes A, Contradictory, or Unrelated.
    """
)

entity_extraction_prompt = PromptTemplate.from_template(
    """Extract entities involved in the following fact, that has been extracted in a text of the type {context}. Only extract the names of standalone entities that are present in the fact, excluding any numbers. Only give entities in a bullet list and do not give any details or explanations.
    
    Fact: {fact}
    
    Entities:
    - """
)

entity_selection_prompt = PromptTemplate.from_template(
    """Here is a fact: {fact}. It has been extracted in a text of the following type: {context}.
    
    Do you think the entity "{entity}" is directly involved in the fact stated above? Please answer Yes or No.
    """
)
#     """
#     Here is a fact: {fact}. It has been extracted in a text of the following type: {context}.
    
#     Among the following entities, which are involved in the fact stated above? Please repeat those you want to keep in a list below. If none of the entities is involved in the fact stated above, please answer None. It is important that you only keep entities among those listed below, and only if they are directly mentioned in the fact stated above (although they may be mentioned in a different form).
    
#     {choices}
#     """
# )


entity_equivalence_prompt = PromptTemplate.from_template(
    """Here is a fact: {fact} It has been extracted in a text of the following type: {context}
    
    Among the following entities, is there one that is equivalent to the entity {entity} involved in the fact stated above? Please answer with the name of the equivalent entity, or None if there is no equivalent entity.
    
    Target entity: {entity}
    
    List of candidates:
    {choices}
    - None of the above
    
    Equivalent entity or None: """
)
    

new_entities_prompt = PromptTemplate.from_template(
    """
    You are an annotator that extracts entities involved in a simple fact. Your task is to select atomistic and standalone concepts that are involved in the fact stated below. For each one, give simply its name. If you think that the fact does not involve any entity, please answer None. Here is the fact:
    
    Fact: {fact}. 
    (Context: {context})
    
    Entities:
    {entity_chunk}"""
)

def get_new_entities_prompt(fact, context, entities):
    if entities:
        entity_chunk = entities=itemize_list(entities)
    else:
        entity_chunk = "-"
        
    return new_entities_prompt.format(fact=fact, context=context, entity_chunk=entity_chunk)


question_prompt = PromptTemplate.from_template(
    """
    {question}
    
    You can answer this question making use of the facts stated below. If one of this facts is particularly relevant to your answer, please mention its number.
    {facts}
    """
)