from langchain.text_splitter import RecursiveCharacterTextSplitter


def split_with_chunk_overlap(documents, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, 
                                                   chunk_overlap=chunk_overlap)
    
    return text_splitter.split_documents(documents)