# Scripts for running experiments

### Structure

The scripts for running experiments are located in this folder.

-  [wikiner.py](./wikiner.py) -- type (NE class) induction experiment on WikiNer corpus.
-  [chi_example](./chi_example.py) -- a small example of type (NE class) induction on [LER](https://github.com/elenanereiss/Legal-Entity-Recognition) corpus.
- [wsid_example](./wsid_example.py) -- a small example to induce word / NE senses, see the 12 contexts with `Jaguar` in the script itself.

Configuration files used by some scripts are located in [configs](./configs).

- `wikiner_*_config.conf` -- configurations to run WikiNer experiments. Description of the variables are provided in the file itself.
- [logging.conf](./configs/logging.conf) -- configuration of logging.

## Run as Docker

You can use the `Dockerfile` and run the scripts in the docker container. For this `cd` to the root folder where the `Dockerfile` is located and run `docker run -it`. By default it will execute `chi_example.py` script, change the last line in the `Dockerfile` toi run a different script. 

## WikiNer experiment

#### to run without linker

Example for English `en`. For German change `en` to `de`.

0. Prepare a virtual environment, install all requirements with `pip3 install -r requirements`.
1. Get WikiNer corpus. Check [the original paper](https://www.sciencedirect.com/science/article/pii/S0004370212000276).
2. Induce senses of Namned Entities annotated in the WikiNer corpus:     
   1. In [wikiner_en_senses_config.conf]('./configs/wikiner_en_senses_config.conf') set `linker=dummy`, set `conll_data_path` to point to the actual WikiNer corpus in .tsv format.
   2. Run
      ```bash
      python3 wikiner_senses configs/wikiner_en_senses_config.conf
      ```
   3. Check the induced sense in `<nes_senses_output_stem>_m<m_value>_k<k_value>.json`, where `*_value` is the respective integer value.  
3. Induce new types from those senses:
   1. In [wikiner_en_types_config.conf]('./configs/wikiner_en_types_config.conf') set `ners_predictions_path` to the output target file with induced senses that you want to use.
   2. Run 
      ```bash
      python3 wikiner_new_types configs/wikiner_en_types_config.conf
      ```
   3. Check the results in at `new_types_output_folder`.
4. Compute the evaluation results.    
   See the config file `wikiner_en_config_VICTOR.conf` and create one for your setup. 
   You must take care of the base_dir variable. 
   Call the resulting file `wikiner_en_config_EVAL.conf`
   1. run
         ```bash
      python3 evaluation/evaluation_pipeline.py configs/wikiner_en_config_EVAL.conf
      ```



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
7. Fill in the [wikiner_en_senses_config.conf]('./configs/wikiner_en_senses_config.conf'):
    1. Set `linker=wikidata`,
    2. Set `el_url=http://localhost:8090`
    3. Set `ner_contexts_output` to some non-existing file. !WARNING! if set to an existing file, the script will attempt to read the file.