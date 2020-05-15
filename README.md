# Pretrained Language Model for WSID

The package implements a WSID method based on clustering predictions from a pre-trained language model.
In the current implementation pre-trained [BERT](https://github.com/google-research/bert) is used. 
For clustering we represent the obtained infromation about predictions of possible substitues as a binary matrix (contexts X substitues). Then we use Algorithm 2 from [https://doi.org/10.1016/j.jcss.2009.05.002] to obtain the binary factors and filter the clusters to obtain the senses.

See also [this blogpost](https://medium.com/@revenkoartem/label-unstructured-data-using-enterprise-knowledge-graphs-2-d84bda281270) for further details.


## Getting Started

### Prerequisites

1. You need to download the required nltk and spacy datasets.

    ```
    python -m spacy download en_core_web_sm de_core_news_sm nl_core_news_sm es_core_news_sm
    ``` 
    
    ```
    python -m nltk.downloader punkt stopwords averaged_perceptron_tagger wordnet
    ```

2. You need to point to the language model that you would like to use. For this populate the value in `.env`. By default `bert-base-multilingual-uncased` is used.

### Installing

Easy install with pip from the github repo:
```bash
pip install -e git://github.com/semantic-web-company/ptlm_wsid.git#egg=ptlm_wsid
```


## Usage

For an example of usage see [`example.py`](ptlm_wsid/example.py).


##

### Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

### Authors

* [**Artem Revenko**](https://github.com/artreven) 

### License

This project is licensed under the MIT License.

### TODOs
