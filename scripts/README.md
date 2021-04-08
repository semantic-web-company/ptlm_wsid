# Scripts for running experiments

### Structure

The scripts for running experiments are located in this folder.

-  [wikiner.py](./wikiner.py) -- type (NE class) induction experiment on WikiNer corpus.
-  [chi_example](./chi_example.py) -- a small example of type (NE class) induction on [LER](https://github.com/elenanereiss/Legal-Entity-Recognition) corpus.
- [wsid_example](./wsid_example.py) -- a small example to induce word / NE senses, see the 12 contexts with `Jaguar` in the script itself.

Configuration files used by some scripts are located in [configs](./configs).

- `wikiner_*_config.conf` -- configurations to run WikiNer experiments. Description of the variables are provided in the file itself.
- [logging.conf](./configs/logging.conf) -- configuration of logging.

## WikiNer experiment

#### to make use of entity linking using wikidata
You will need to download and setup an [Entity Fishing](https://nerd.readthedocs.io/en/latest/) service. For this you can follow:

1. Clone the git repo `https://github.com/syats/entity-fishing.git`
2. Go into that repo's main directory `cd entity-fishing`
3. Run:  `docker build -t fishing ./docker/`
4. Meanwhile download the very heavy database that Entity Fishing uses
```
 cd entity-fishing/data/db
 wget https://science-miner.s3.amazonaws.com/entity-fishing/0.0.3/linux/db-de.zip 
 wget https://science-miner.s3.amazonaws.com/entity-fishing/0.0.3/linux/db-en.zip 
 wget https://science-miner.s3.amazonaws.com/entity-fishing/0.0.3/linux/db-kb.zip 
 unzip db-de.zip
 unzip db-kb.zip
 mv *zip ..

```  
5. Run container:
 ` cd ../..`
 ` docker run --rm --name fishing_ctr -p 8090:8090 -v ${PWD}/data/db:/fishing/nerd/data/db/ -it fishing`
6. Enjoy at `http://localhost:8090/`.

   Example: `curl 'http://localhost:8090/service/disambiguate' -X POST -F "query={ 'text': 'Die Schlacht bei Tannenberg war eine Schlacht des Ersten Weltkrieges und fand in der Gegend südlich von Allenstein in Ostpreußen vom 26. August bis 30. August 1914 zwischen deutschen und russischen Armeen statt. Die deutsche Seite stellte hierbei 153.000 Mann, die russische Seite 191.000 Soldaten ins Feld. Sie endete mit einem Sieg der deutschen Truppen und der Zerschlagung der ins südliche Ostpreußen eingedrungenen russischen Kräfte.', 'language':'de'}"`

#### to run 

`python3 wikiner.py configs/wikiner_en_config.conf` -- to run with English WikiNer.