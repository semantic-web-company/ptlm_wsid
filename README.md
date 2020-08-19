# Pretrained Language Model for WSID

The package implements a WSID method based on clustering predictions from a pre-trained language model.
In the current implementation the pre-trained multilingual
  [DistilBERT](https://huggingface.co/transformers/model_doc/distilbert.html) model is used.
For clustering we represent information as binary table: entities or contexts X predictions by LM. Then we use modified Algorithm 2 from [https://doi.org/10.1016/j.jcss.2009.05.002](https://doi.org/10.1016/j.jcss.2009.05.002) to obtain the binary factors and filter the clusters to obtain the senses.

See also [this blogpost](https://medium.com/@revenkoartem/label-unstructured-data-using-enterprise-knowledge-graphs-2-d84bda281270) for further details.


## Getting Started

### Prerequisites

You need to download the required nltk datasets.
    
```bash
    python -m nltk.downloader punkt stopwords averaged_perceptron_tagger wordnet
```

### Installing

Easy install with pip from the github repo:
```bash
pip install -e git://github.com/semantic-web-company/ptlm_wsid.git#egg=ptlm_wsid
```


## Usage

For an example of usage of WSID see [`scripts/wsid_example.py`](scripts/wsid_example.py).

For an example of usage class hierarchy induction see [`scripts/chi_example.py`](scripts/chi_example.py).

For repeating WiC-TSV experiment see [`scripts/wic_tsv.py`](scripts/wic_tsv.py). Do not forget to define the necessary env variables `WIC_TSV_TRAIN_PATH` and `WIC_TSV_TEST_PATH`. 


### Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/semantic-web-company/ptlm_wsid/tags).

### Authors

* [**Artem Revenko**](https://github.com/artreven) 

### License

This project is licensed under the MIT License.

### TODOs
