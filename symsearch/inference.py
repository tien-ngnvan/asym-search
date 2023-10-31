import torch
import logging
from typing import Union, List, Dict
from transformers import AutoTokenizer, AutoConfig, AutoModel
 

logger = logging.getLogger(__name__)



class RetrieveInference:
    def __init__(self, 
                 model_name_or_path:str = None,
                 q_max_length:int = 96,
                 p_max_length:int = 256,
                 device:str = None) -> None:
        
        """
            Args:
                model_name_or_path: name of model load from HUB
                q_max_length: input query max length tokenized
                p_max_length: input passage max length tokenized
                device: determine which process to run. Default is cpu
        """
        if device is not None:
            try:
                self.device = torch.device(device)
            except:
                logger.error("Set device fail. Get default cpu")
                self.device = torch.device('cpu')
        else:
            if torch.cuda.is_available():
                self.device = torch.device('cuda:0') if torch.cuda.device_count() > 1 else torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        
        self.q_max_length = q_max_length
        self.p_max_length = p_max_length
        
        # Load model directly
        self.config = AutoConfig.from_pretrained(model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path,config=self.config).to(self.device)
        
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
    
    def get_tokenized(self, sentences:str , max_length:int = 256) -> Dict[str, torch.Tensor]:
        """ Encode string to token ids. 

        Args:
            sentences (str): _description_
            max_length (int): _description_

        Returns:
            torch.Tensor: _description_
        """
        tokenized = self.tokenizer(sentences, max_length=max_length,
                                   padding=True, truncation=True,
                                   return_tensors='pt').to(self.device)
        
        return tokenized


class RerankInference:
    def __init__(self, 
                 model_name_or_path:str = None,
                 q_max_length:int = 96,
                 p_max_length:int = 256,
                 device:str = None) -> None:
        """
            Args:
                model_name_or_path: name of model load from HUB
                q_max_length: input query max length tokenized
                p_max_length: input passage max length tokenized
                device: determine which process to run. Default is cpu
        """
        if device is not None:
            try:
                self.device = torch.device(device)
            except:
                logger.error("Set device fail. Get default cpu")
                self.device = torch.device('cpu')
        else:
            if torch.cuda.is_available():
                self.device = torch.device('cuda:0') if torch.cuda.device_count() > 1 else torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        
        self.q_max_length = q_max_length
        self.p_max_length = p_max_length
        
        # Load model directly
        self.config = AutoConfig.from_pretrained(model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path,config=self.config).to(self.device)
        
    def encode_pair(self, query:str=None, passage:str = None ) -> float:
        """This function take query and passage as pair to calculate similarity score

        Args:
            query (str, optional): sentence using as a question
            passage (str, optional): sentence 
        Returns:
            float: score
        """
        assert isinstance(query, str), f"Input query should be a string not a {type(query)}"
        assert isinstance(passage, str), f"Input passage should be a string not a {type(passage)}." \
                                        + "If you want to run with batch, use encode_pairs"
 
        tokenized = self.get_tokenized(query, passage, 
                                       max_length= self.q_max_length + self.p_max_length)
        with torch.no_grad:
            model_output = self.model(pair=tokenized)
            score = model_output.logits.cpu().detach().numpy()
        
        #  multiply by -1 since score return negative iner product
        return score[0] * -1
    
    def encode_pairs(self, query:str=None, passages:List[str] = None ) -> List[float]:
        """This function take query and passage as pair to calculate similarity score

        Args:
            query (str, optional): sentence using as a question
            passage (str, optional): sentence 
        Returns:
            float: List of scores
        """
        
        if isinstance(query, str):
            num_pg = len(passages)
            query = [query] * num_pg
        assert len(query) == len(passages), f'Number of queries not the same number of passages'

        tokenized = self.get_tokenized(query, passages, 
                                       max_length= self.q_max_length + self.p_max_length)
        with torch.no_grad():
            ranker_logits = self.model(tokenized).logits
            scores = ranker_logits.cpu().detach().numpy()
        
        #  multiply by -1 since score return negative iner product
        return [item[0]*-1 for item in scores]
    
    def get_tokenized(self, s1: str, s2: str, max_length:int = 368)-> Dict[str, torch.Tensor]:
        """Encode string to token ids 

        Args:
            s1 (str): sentence 1 
            s2 (str): sentence 2
            max_length (int, optional): max length encode two sentences. Defaults to 368.

        Returns:
            Dict[str, torch.Tensor]: torch.Tensor 
        """
        tokenized = self.tokenizer(s1, s2, max_length=max_length,
                                   padding='max_length', truncation=True,
                                   return_attention_mask=False, return_token_type_ids=True,
                                   return_tensors='pt').to(self.device)
        
        return tokenized