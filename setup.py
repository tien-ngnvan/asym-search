from setuptools import setup, find_packages

setup(
    name='asym-search',
    version='0.0.1',
    package_dir={"": "asym_search"},
    packages=find_packages("asym_search"),
    url='https://github.com/tien-ngnvan/asym-search',
    author='tien-ngnvan',
    author_email='tien.ngnvan@gmail.com',
    description='Asym-search is a search engine for research and development that uses a searching and ranking pipeline in Dense Passage Retrieval.',
    python_requires='>=3.8',
    zip_safe=False,
    install_requires=[
        "transformers>=4.28"
    ],
    
)