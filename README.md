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

#### to make use of entity linking using wikidata
You will need to download and setup an [Entity Fishing](https://nerd.readthedocs.io/en/latest/) service. For this you can follow:

1. Create a new directory, in that directory:
2. Clone the git repo `https://github.com/syats/entity-fishing.git`
3. Go into that repo's main directory, `[repo_root]`, 
4. do:  `docker build -t fishing .`
5. Meanwhile download the very heavy database that Entity Fishing uses
```
 cd [repo-root]/data/db
 wget https://science-miner.s3.amazonaws.com/entity-fishing/0.0.3/linux/db-de.zip 
 wget https://science-miner.s3.amazonaws.com/entity-fishing/0.0.3/linux/db-en.zip 
 wget https://science-miner.s3.amazonaws.com/entity-fishing/0.0.3/linux/db-kb.zip 
 unzip db-de.zip
 unzip db-kb.zip
 mv *zip ..

```  
7. Run container:
 ` cd [repo-root]`
 ` docker run --rm --name fishing_ctr -p 8090:8090 -v ${PWD}/data/db:/fishing/nerd/data/db/ -it fishing`
8. Enjoy in `http://localhost:9897/`
  Example: `curl 'http://localhost:8090/service/disambiguate' -X POST -F "query={ 'text': 'Die Schlacht bei Tannenberg war eine Schlacht des Ersten Weltkrieges und fand in der Gegend südlich von Allenstein in Ostpreußen vom 26. August bis 30. August 1914 zwischen deutschen und russischen Armeen statt. Die deutsche Seite stellte hierbei 153.000 Mann, die russische Seite 191.000 Soldaten ins Feld. Sie endete mit einem Sieg der deutschen Truppen und der Zerschlagung der ins südliche Ostpreußen eingedrungenen russischen Kräfte.', 'language':'de'}"`


### Installing

Easy install with pip from the github repo:
```bash
pip install -e git://github.com/semantic-web-company/ptlm_wsid.git#egg=ptlm_wsid
```


## Usage

For an example of usage of WSID see [`scripts/wsid_example.py`](scripts/wsid_example.py).

For an example of usage class hierarchy induction see [`scripts/chi_example.py`](scripts/chi_example.py).


### Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/semantic-web-company/ptlm_wsid/tags).

### License

This project is licensed under the MIT License.
