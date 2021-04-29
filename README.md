# Pretrained Language Model for WSID

The package implements a WSID method based on clustering predictions from a pre-trained language model.
In the current implementation the pre-trained multilingual
  [DistilBERT](https://huggingface.co/transformers/model_doc/distilbert.html) model is used.
For clustering we represent information as binary table: entities or contexts X predictions by LM. Then we use modified Algorithm 2 from [https://doi.org/10.1016/j.jcss.2009.05.002](https://doi.org/10.1016/j.jcss.2009.05.002) to obtain the binary factors and filter the clusters to obtain the senses.

See also [this blogpost](https://medium.com/@revenkoartem/label-unstructured-data-using-enterprise-knowledge-graphs-2-d84bda281270) for further details.


## Getting Started

[comment]: <> (### Prerequisites)

[comment]: <> (You need to download the required nltk datasets.)
    
[comment]: <> (```bash)

[comment]: <> (    python -m nltk.downloader punkt stopwords averaged_perceptron_tagger wordnet)

[comment]: <> (```)


### Installing

Easy install with pip from the github repo:
```bash
pip install -e git://github.com/semantic-web-company/ptlm_wsid.git#egg=ptlm_wsid
```


## Usage

Find examples of usage in [scripts](./scripts).


### Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/semantic-web-company/ptlm_wsid/tags).

### License

This project is licensed under the MIT License.
