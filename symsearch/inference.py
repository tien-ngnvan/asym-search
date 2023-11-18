import torch
from typing import Union, List
from transformers import AutoTokenizer, AutoConfig

from .retrieve.modeling import RetrieveModel
import logging

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)


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
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = RetrieveModel.load(
            model_name_or_path,
            config=AutoConfig.from_pretrained(model_name_or_path)).to(self.device)
        
        logger.info(self.model.eval())
        
    def embed_query(self, sentences:Union[str, List[str]] = None, convert_to_list=True):
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

        if convert_to_list:
            if embeddings.shape[0] == 1:
                return list(embeddings.reshape(-1))
            else:
                return [list(embd.reshape(-1)) for embd in embeddings]
            
        return embeddings

    def embed_documents(self, sentences:Union[str, List[str]] = None, convert_to_list=True):
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

        if convert_to_list:
            if embeddings.shape[0] == 1:
                return list(embeddings.reshape(-1))
            else:
                return [list(embd.reshape(-1)) for embd in embeddings]
            
        return embeddings 
    
    def get_tokenized(self, sentences, max_length):
        tokenized = self.tokenizer(sentences, max_length=max_length,
                                   padding=True, truncation=True,
                                   return_tensors='pt').to(self.device)
        
        return tokenized