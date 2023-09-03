import torch
from typing import Union, List
from transformers import AutoTokenizer, AutoConfig

from retrieve.modeling import RetrieveModel
from rerank.modeling import RerankModel

import logging

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)


class RetrieveInference:
    def __init__(self, 
                 model_name_or_path:str = None,
                 q_max_length:int = 96,
                 p_max_length:int = 384,
                 device_type:str = 'cpu') -> None:
        
        """
            Args:
                model_name_or_path: name of model load from HUB
                q_max_length: input query max length tokenized
                p_max_length: input passage max length tokenized
                device_type: determine which process to run. Default is cpu
        """
        
        if device_type == 'cpu':
            self.device = torch.device('cpu')
        elif device_type == 'gpu':
            if torch.cuda.is_available():
                self.device = torch.device('cuda:0')
            else:
                logger.info('There is no cuda available with setting device_type = gpu')
                self.device = torch.device('cpu')
        
        self.q_max_length = q_max_length
        self.p_max_length = p_max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = RetrieveModel.load(
            model_name_or_path,
            config=AutoConfig.from_pretrained(model_name_or_path)).to(self.device)
        
        logger.info(self.model.eval())
        
    def encode_question(self, sentences:Union[str, List[str]] = None) -> torch.Tensor:
        """This function encode a question/query to embedding vector

        Args:
            sentences (Union[str, List[str]]): _description_
            tokenizer (_type_): _description_
            device (_type_): _description_

        Returns:
            torch.Tensor: embedding(s) of input sentences
        """
        tokenized = self.get_tokenized(sentences=sentences, max_length=self.q_max_length)
        with torch.no_grad():
            model_output = self.model(query=tokenized)
            embeddings = model_output.q_reps.cpu().detach().numpy()  
            
        return embeddings

    def encode_context(self, sentences:Union[str, List[str]] = None):
        """This function encode a passage/context to embedding vector

        Args:
            sentences (Union[str, List[str]], optional): a string or list of string to be input encode model

        Returns:
            torch.Tensor: embedding(s) of input sentences
        """
        tokenized = self.get_tokenized(sentences=sentences, max_length=self.p_max_length)
        with torch.no_grad():
            model_outputs = self.model(passage=tokenized)
            embeddings = model_outputs.p_reps.cpu().detach().numpy()  
            
        return embeddings 
    
    def get_tokenized(self, sentences, max_length):
        tokenized = self.tokenizer(sentences, max_length=max_length,
                                   padding=True, truncation=True,
                                   return_tensors='pt').to(self.device)
        
        return tokenized


class RerankInference:
    def __init__(self, 
                 model_name_or_path:str = None,
                 q_max_length:int = 96,
                 p_max_length:int = 384,
                 device_type:str = 'cpu') -> None:
        """
            Args:
                model_name_or_path: name of model load from HUB
                q_max_length: input query max length tokenized
                p_max_length: input passage max length tokenized
                device_type: determine which process to run. Default is cpu
        """
        
        if device_type == 'cpu':
            self.device = torch.device('cpu')
        elif device_type == 'gpu':
            if torch.cuda.is_available():
                self.device = torch.device('cuda:0')
            else:
                logger.info('There is no cuda available with setting device_type = gpu')
                self.device = torch.device('cpu')
        
        self.q_max_length = q_max_length
        self.p_max_length = p_max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = RerankModel.load(
            model_name_or_path,
            config=AutoConfig.from_pretrained(model_name_or_path)).to(self.device)
        
        logger.info(self.model.eval())
        
    def encode_pair(self, query:str=None, passage:str = None ) -> float:
        """This function take query and passage as pair to calculate similarity score

        Args:
            query (str, optional): sentence using as a question
            passage (str, optional): sentence 
        Returns:
            float: score
        """
        assert isinstance(query, str), f"Input query should be a string not a {type(query)}"
        assert isinstance(passage, str), f"Input passage should be a string not a {type(passage)}"
 
        tokenized = self.tokenizer(query.strip(), passage.strip(),
                                   max_length= self.q_max_length + self.p_max_length,
                                   truncation='only_first', padding='max_length',
                                   return_attention_mask=False, return_token_type_ids=True,
                                   return_tensors='pt')
        
        model_output = self.model(pair=tokenized)
        score = model_output.scores.cpu().detach().numpy()[0]
        
        return score


if __name__ == '__main__':      
    
    # test retrieval
    retrieval = RetrieveInference(
        model_name_or_path="caskcsg/cotmae_base_msmarco_retriever",
        q_max_length=128,
        p_max_length=384,
        device_type='cpu'
    )
    text = ["I have to go there"] * 5
    encode = retrieval.encode_context(text)
    print(encode)
    
    # test rerank
    rerank = RerankInference(
        model_name_or_path="caskcsg/cotmae_base_msmarco_reranker",
        q_max_length=128,
        p_max_length=384,
        device_type='cpu'
    )
    query = "If I hypothetically built a fully functioning rocket and were able to "\
            "fund the trip myself, would it be legal for me to leave earth?"
    passage = "crew, who have become the first humans to travel into space. The rocket is at first thought to be lost, " \
            "having dramatically overshot its planned orbit, but eventually it is detected by radar and returns to Earth, " \
            "crash-landing in Wimbledon, London.\nWhen Quatermass and his team reach the crash area and succeed in opening the rocket, " \
            "they discover that only one of the three crewmen, Victor Carroon, remains inside. Quatermass and his chief assistant Paterson " \
            "(Hugh Kelly) investigate the rocket's interior and are baffled by what they find: the space suits of the others are present, and the instruments on board"
    score = rerank.encode_pair(query=query, passage=passage)[0]
    print(score)