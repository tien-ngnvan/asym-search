from setuptools import setup, find_packages

with open("README.md", 'r') as f: 
    long_description= f.read() 
    

setup(
    name='symsearch',
    version='0.0.2',
    packages=find_packages(),
    url='https://github.com/tien-ngnvan/symsearch',
    author='tien-ngnvan',
    author_email='tien.ngnvan@gmail.com',
    description='symsearch is a search engine for research and development that uses a searching and ranking pipeline in Dense Passage Retrieval.',
    long_description= long_description, 
    long_description_content_type= 'text/markdown',
    maintainer='Tien Nguyen Van',
    maintainer_email= 'tien.ngnvan@gmail.com',
    python_requires='>=3.8',
    zip_safe=False,
    install_requires=[
        "transformers"
    ],
    
)