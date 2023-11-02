from langchain.document_loaders import PyPDFDirectoryLoader


def load_pdf_files(datapath):
    loader = PyPDFDirectoryLoader(datapath)
    
    return loader.load()


def load_text_files(datapath):
    with open(datapath, 'r', encoding='utf-8') as file:
        data = file.readline()
        
    return data
