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

reformulation_prompt = PromptTemplate.from_template(
    """You are a knowledge annotator that extracts standalone facts from text. Your task is to reformulate the following fact in a more general way, so that it can be understood without the context of the text. 
    
    Here is the fact: {fact}. 
    
    It has been extracted in a text of the following type: {context}. The entire context is the following snippet:
    
    {text}
    
    Please reformulate the fact so that it can be understood without the context of the text, without changing its meaning. In particular, remove all pronouns and references to elements not in the fact. If the fact is already general enough, keep it as it is. DO NOT ADD ANY INFORMATION, STICK TO THE FACT AS IT IS STATED. Make sure the reformulated fact is atomic and as concise as possible.
    
    Reformulated fact: """
)

confrontation_prompt = PromptTemplate.from_template(
    """You are a knowledge extractor that identifies overlapping facts. You have identified a new fact and you want to check if it is already present in the database. Here is the new fact: {fact}. The context in which it was extracted is the following snippet: {context}. 
    
    In the database, we have found this similar fact: {other_fact}. Do you think the second fact is redundant with the first one? Answer Yes or No.
    """
)

entity_extraction_prompt = PromptTemplate.from_template(
    """Extract entities involved in the following fact. Only extract the names of standalone entities that are explicitly stated in the fact, and exclude any numbers. Only give the names of entities in a bullet list so that they can be recognized, and do not explain any thing about them. In particular, do not mention why you added the entity, and do not mention if it is implicit or explicit.
    
    Fact: {fact}
    
    Entities:
    - """
)

entity_selection_prompt = PromptTemplate.from_template(
    """Here is a fact: {fact}.
    
    Do you think the entity "{entity}" is directly and explicitly involved in the fact stated above? You should answer yes only if the entity is very explicitly present in the sentence that states the fact. Please answer Yes or No.
    """
)

entity_equivalence_prompt = PromptTemplate.from_template(
    """Are the entities {entity} and {other} equivalent? Please answer Yes or No."""
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


retrieval_prompt = PromptTemplate.from_template(
    """
    {query}
    
    You may answer using the following facts if they are useful (simply ignore irrelevant facts without mentioning them):
    {facts}
    """
)