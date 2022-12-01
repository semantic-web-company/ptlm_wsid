To run any of these scripts follow "run as Docker" from the main README and change the last line starting from `ENTRYPOINT` in the `Dockerfile` to point to the respective script.

# Word Sense Induction example

[./wsid_example](./wsid_example.py) -- a small example to induce word / NE senses, see the 12 contexts with `Jaguar` in the script itself. The script does not require any external input, all the necessary data -- small corpus of 12 context -- is included in the script itself. The results are printed to the terminal. For more details see [this blogpost](https://medium.com/@revenkoartem/label-unstructured-data-using-enterprise-knowledge-graphs-2-d84bda281270).

# Class Hierarchy Induction example

[./chi_example](./chi_example.py) -- a small example of class (Named Entity type) induction on [LER](https://github.com/elenanereiss/Legal-Entity-Recognition) corpus. The script downloads the LER dataset and is, therefore, self-contained. The results are printed to the terminal. For more details see [this blogpost](TO BE ADDED LATER). 

# WikiNER experiment

What follows contains step by step instructions to repeat experiments on WikiNER dataset with evaluations. The experiments are describedin further details in [this paper](https://www.researchgate.net/publication/363368922_Learning_Ontology_Classes_from_Text_by_Clustering_Lexical_Substitutes_Derived_from_Language_Models1).

_____
Configuration files are located in [./configs](./configs):
- `./configs/wikiner_*_config.conf` -- configurations to run WikiNer experiments. Description of the variables are provided in the file itself.
- [./configs/logging.conf](./configs/logging.conf) -- configuration of logging.


#### to run without linker

Example for English `en`. For German change `en` to `de`.

0. Prepare a virtual environment, install all requirements with `pip3 install -r requirements`.
1. Get WikiNer corpus. Check [the original paper](https://www.sciencedirect.com/science/article/pii/S0004370212000276).
2. Induce senses of Namned Entities annotated in the WikiNer corpus:     
   1. In [wikiner_en_senses_config.conf]('./configs/wikiner_en_senses_config.conf') set `linker=dummy`, set `conll_data_path` to point to the actual WikiNer corpus in .tsv format.
   2. Run
      ```bash
      python3 wikiner1_senses configs/wikiner_en_senses_config.conf
      ```
   3. Check the induced sense in `<nes_senses_output_stem>_m<m_value>_k<k_value>.json`, where `*_value` is the respective integer value.  
3. Induce new types from those senses:
   1. In [wikiner_en_types_config.conf]('./configs/wikiner_en_types_config.conf') set `ners_predictions_path` to the output target file with induced senses that you want to use.
   2. Run 
      ```bash
      python3 wikiner2_new_types configs/wikiner_en_types_config.conf
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