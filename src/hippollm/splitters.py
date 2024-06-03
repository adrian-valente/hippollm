from typing import Literal, Optional
import abc
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

## Internal types

TSplitStrategyLiteral = Literal['naive', 'paragraph', 'recursive', 'semantic']
    
@dataclass
class Chunk:
    text: str
    pos: tuple[int, int]    
    

## Helpers

def get_chunks_with_positions(text: str, _chunks: list[str]) -> list[tuple[int, int]]:
    """Get the positions of the chunks in the text."""
    chunks = []
    i = 0
    for chunk in _chunks:
        i += text[i:].find(chunk[:min(100, len(chunk))])
        pos = (i, i + len(chunk))
        chunks.append(Chunk(text=chunk, pos=pos))
    return chunks


## Text splitters

class TextSplitter(abc.ABC):
    """Base class for text splitters."""
    
    @abc.abstractmethod
    def __init__(self, 
                 chunk_size: Optional[int] = None,
                 **kwargs) -> None:
        raise NotImplementedError
    
    @abc.abstractmethod
    def split(self, text: str) -> list[Chunk]:
        raise NotImplementedError
    

class NaiveTextSplitter(TextSplitter):
    """Just break in chunks"""
    
    def __init__(self, 
                 chunk_size: Optional[int] = None,
                 **kwargs) -> None:  
        if chunk_size is None:
            raise ValueError("Chunk size must be provided for naive splitting.")
        self.chunk_size = chunk_size
        
    def split(self, text: str) -> list[Chunk]:  
        chunks = []
        for i in range(0, len(text), self.chunk_size):
            chunk = text[i: min(i+self.chunk_size, len(text))]
            chunks.append(Chunk(text=chunk, pos=(i, i+len(chunk))))
        return chunks
            
    
class RecursiveTextSplitter(TextSplitter):
    """Text splitter based on the RecursiveCharacterTextSplitter."""
    
    def __init__(self,
                 chunk_size: Optional[int] = None,
                 **kwargs) -> None:
        if chunk_size is None:
            raise ValueError("Chunk size must be provided for recursive splitting.")
        separators = ['\n\n', '\n', '. ', '? ', '! ', '; ', ', ', ' ']
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, separators=separators)
    
    def split(self, text: str) -> list[Chunk]:
        _chunks = self.splitter.split_text(text)
        return get_chunks_with_positions(text, _chunks)


class ParagraphTextSplitter(TextSplitter):
    """Just break in paragraphs"""
    
    def __init__(self,
                 chunk_size: Optional[int] = None,
                 **kwargs) -> None:
        pass
        
    def split(self, text: str) -> list[Chunk]:
        _chunks = text.split('\n\n')
        i = 0
        chunks = []
        for chunk in _chunks:
            pos = (i, i + len(chunk))
            chunks.append(Chunk(text=chunk, pos=pos))
            i += len(chunk) + 2
        return chunks
  
    
class SemanticTextSplitter(TextSplitter):
    
    def __init__(self,
                 chunk_size: Optional[int] = None,
                 **kwargs) -> None:
        try:
            from langchain_experimental.text_splitter import SemanticChunker
        except ImportError:
            raise ImportError("Semantic text splitter requires langchain_experimental. "
                              "Install it with 'pip install langchain_experimental'.")
        from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
        if 'embedding_model' in kwargs:
            self.embedding_model_name = kwargs['embedding_model']
        else:
            self.embedding_model_name = 'all-MiniLM-L6-v2'
        self.embeddings = SentenceTransformerEmbeddings(model_name=self.embedding_model_name)
        self.splitter = SemanticChunker(embeddings=self.embeddings)
        
    def split(self, text: str) -> list[Chunk]:
        _chunks = self.splitter.split_text(text)  
        return get_chunks_with_positions(text, _chunks)
        

def get_splitter(strategy: TSplitStrategyLiteral, 
                 chunk_size: Optional[int] = None,
                 **kwargs) -> TextSplitter:
    """Get the appropriate splitter based on the strategy."""
    if strategy == 'naive':
        return NaiveTextSplitter(chunk_size, **kwargs)
    elif strategy == 'paragraph':
        return ParagraphTextSplitter(chunk_size, **kwargs)
    elif strategy == 'recursive':
        return RecursiveTextSplitter(chunk_size, **kwargs)
    elif strategy == 'semantic':
        return SemanticTextSplitter(chunk_size, **kwargs)
    else:
        raise ValueError(f"Unknown text splitting strategy: {strategy}")
    