# ASYM-SEARCH
Asym-search is a search engine for research and development that uses a searching and ranking pipeline in Dense Passage Retrieval. 

## Installation
Asym-search recommends Python 3.8 or higher version. 

Install with ``pip``
```
pip install asym-search
```

## Training
    Updating . . .
## Inference 
### Retrieval
1. Query embedding
  
``` python
from asym_search import RetrieveInference

sentence = "Why does water heated to room temperature feel colder than the air around it?"

retrieve = RetrieveInference(
        model_name_or_path="caskcsg/cotmae_base_msmarco_retriever",
        q_max_length=128,
        device_type='cpu'
      )
query_embd = retrieval.encode_question(sentence)
print(query_embd)
```
2. Passage embedding
``` python
from asym_search import RetrieveInference

sentence = "Water transfers heat more efficiently than air. When something feels cold it's " \
          "because heat is being transferred from your skin to whatever you're touching. " \
          "Since water absorbs the heat more readily than air, it feels colder."

retrieve = RetrieveInference(
        model_name_or_path="caskcsg/cotmae_base_msmarco_retriever",
        p_max_length=384,
        device_type='cpu'
      )
passage_embd = retrieval.encode_context(sentence)
print(passage_embd)
```
    
### Reranker
```python
from asym_search import RerankInference

sentence1 = "If I hypothetically built a fully functioning rocket and were able to "\
            "fund the trip myself, would it be legal for me to leave earth?"
sentence2 = "crew, who have become the first humans to travel into space. The rocket is at first thought to be lost, " \
            "having dramatically overshot its planned orbit, but eventually it is detected by radar and returns to Earth, " \
            "crash-landing in Wimbledon, London.\nWhen Quatermass and his team reach the crash area and succeed in opening " \
            "the rocket, they discover that only one of the three crewmen, Victor Carroon, remains inside."

rerank = RerankInference(
        model_name_or_path="caskcsg/cotmae_base_msmarco_reranker",
        q_max_length=128,
        p_max_length=384,
        device_type='cpu'
      )
score = rerank.encode_pair(query=sentence1, passage=sentence2)[0]
print(score)
```

# Contacts
If you have any questions/suggestions feel free to open an issue or send general ideas through email.
- Contact person: tien.ngnvan@gmail.com
  
> This repository contains experimental research and developments purpose of giving additional background details on Dense Passage Retrieval.

