from abc import ABC, abstractmethod
from typing import Any, List, Optional
from overrides import override
import os


class LlmBackend(ABC):
    client: Any
    handles_grammars: bool = False
    
    @abstractmethod
    def invoke(self, 
               prompt: str, 
               optional_grammar: Optional[str] = None,
               max_tokens: Optional[int] = None,
               stop: Optional[List[str]] = None,
               ) -> str:
        raise NotImplementedError
    
    
class LlmBackendOllama(LlmBackend):
    def __init__(self, model: str, **params):
        from langchain_community.llms import Ollama
        try:
            self.client = Ollama(model=model)
            self.client.invoke(prompt='Hi! ')
        except Exception as e:
            raise ValueError(f"Could not initialize Ollama with model {model}: {e}")
    
    @override
    def invoke(self, 
               prompt: str, 
               optional_grammar: Optional[str] = None,
               max_tokens: Optional[int] = None,
               stop: Optional[List[str]] = None,
               ) -> str:
        return self.client.invoke(prompt, stop=stop)
    
    
class LlmBackendCpp(LlmBackend):
    def __init__(self, model: str, **params):
        from llama_cpp import Llama
        
        self.handles_grammars = True
        if 'n_ctx' not in params:
            params['n_ctx'] = 5000
        self.chat_model = ('chat_model' in params) and (params['chat_model'] is not None)
        if 'system_prompt' in params:
            self.system_prompt = params['system_prompt']
            del params['system_prompt']
        self.grammars_cache = {} 
            
        try:
            if model.startswith(('/', './')):
                self.client = Llama(model_path=model, **params)
            else:
                self.client = Llama.from_pretrained(model, **params)
            self.client.create_completion('Hi! ', max_tokens=1)
        except Exception as e:
            raise ValueError(f"Could not initialize Llama with model {model}: {e}")
      
    @override  
    def invoke(self, 
               prompt: str, 
               optional_grammar: Optional[str] = None,
               max_tokens: Optional[int] = None,
               stop: Optional[List[str]] = None,
               ) -> str:
        if optional_grammar is not None:
            if optional_grammar in self.grammars_cache:
                grammar = self.grammars_cache[optional_grammar]
            else:
                from llama_cpp import LlamaGrammar
                grammar = LlamaGrammar.from_string(optional_grammar)
                self.grammars_cache[optional_grammar] = grammar
        else:
            grammar = None
        if stop is None:
            stop = []
        
        if self.chat_model:
            messages = [{'role': 'user', 'content': prompt}]
            if hasattr(self, 'system_prompt'):
                messages.insert(0, {'role': 'system', 'content': self.system_prompt})
            outp = self.client.create_chat_completion(
                messages, 
                grammar=grammar, 
                max_tokens=max_tokens,
                stop=stop
            )
            return outp['choices'][0]['message']['content']
        
        else:
            outp = self.client.create_completion(
                prompt, 
                grammar=grammar, 
                max_tokens=max_tokens, 
                stop=stop
            )
            return outp['choices'][0]['text']


class LlmBackendGroq(LlmBackend):
    
    def __init__(self, model: str, **params):
        self.handles_grammars = False
        self.model = model
        
        try:
            from langchain_groq import ChatGroq
        except:
            raise ValueError("You need to install langchain-groq to use this backend: `pip install langchain-groq`")
        
        try:
            groq_api_key = os.environ['GROQ_API_KEY']
        except KeyError:
            raise ValueError("You need to setup your Groq API key with `export GROQ_API_KEY=<your-api-key-here>`")
        
        try:
            self.client = ChatGroq(model_name=model, api_key=groq_api_key, **params)
        except Exception as e:
            raise ValueError(f"Could not reach Groq API: {e}")
            
        
    def invoke(self, 
               prompt: str, 
               optional_grammar: Optional[str] = None,
               max_tokens: Optional[int] = None,
               stop: Optional[List[str]] = None,
               ) -> str:
        return self.client.invoke(prompt, max_tokens=max_tokens, stop=stop).content
    

class LlmBackendOpenAI(LlmBackend):
    
    def __init__(self, model: str, **params):
        self.handles_grammars = False
        self.model = model
        
        try:
            from langchain_openai import ChatOpenAI
        except:
            raise ValueError("You need to install langchain_openai to use this backend: `pip install langchain_openai`")
        
        try:
            self.client = ChatOpenAI(model=model, openai_api_key=os.environ['OPENAI_API_KEY'], **params)
        except KeyError:
            raise ValueError("You need to setup your OpenAI API key with `export "
                             "OPENAI_API_KEY=<your-api-key-here>`")
        except Exception as e:
            raise ValueError(f"Could not initialize OpenAI with model {model}: {e}")
        
    def invoke(self, 
               prompt: str, 
               optional_grammar: Optional[str] = None,
               max_tokens: Optional[int] = None,
               stop: Optional[List[str]] = None,
               ) -> str:
        return self.client.invoke(prompt, max_tokens=max_tokens, stop=stop).content
            
                
def load_llm(model: str, backend: str, **params) -> LlmBackend:
    """Factory function for the LLM backend."""
    if backend == 'ollama':
        return LlmBackendOllama(model, **params)
    elif backend == 'llama-cpp':
        return LlmBackendCpp(model, **params)
    elif backend == 'groq':
        return LlmBackendGroq(model, **params)
    elif backend == 'openai':
        return LlmBackendOpenAI(model, **params)
    else:
        raise ValueError(f"Unknown LLM backend: {backend}")
    