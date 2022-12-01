
# Pretrained Language Model for WSID and Ontology Learning

The package implements routines using pre-trained language models ([BERT](https://en.wikipedia.org/wiki/BERT_(language_model)) and [DistilBERT](https://arxiv.org/abs/1910.01108) in particular) to perform (1) [Word Sense Induction](https://en.wikipedia.org/wiki/Word-sense_induction) and (2) [induction of a hierarchy of classes](https://en.wikipedia.org/wiki/On[to(https://en.wikipedia.org/wiki/Formal_concept_analysis)logy_learning) for the annotated words/phrases. Both methods first make use of the mentioned models to produce [contextual substitutes](https://en.wikipedia.org/wiki/Lexical_substitution) for the given annotated entities in the corpus. Then these substitute are grouped using [factorization of binary matrices](https://en.wikipedia.org/wiki/Matrix_decomposition), in particular [Formal Concept Analysis methodology](https://en.wikipedia.org/wiki/Formal_concept_analysis). Further information can be found in these papers and blogposts:

1. ``` 
   @incollection{revenko2022learning,
    title={Learning Ontology Classes from Text by Clustering Lexical Substitutes Derived from Language Models},
    author={Revenko, Artem and Mireles, Victor and Breit, Anna and Bourgonje, Peter and Moreno-Schneider, Julian and Khvalchik, Maria and Rehm, Georg},
    booktitle={Towards a Knowledge-Aware AI},
    pages={155--169},
    year={2022},
    publisher={IOS Press}
   }```

2. ```
   @InProceedings{10.1007/978-3-030-27684-3_22,
    author="Revenko, Artem and Mireles, Victor",
    editor="Anderst-Kotsis, Gabriele and Tjoa, A Min and Khalil, Ismail and Elloumi, Mourad and Mashkoor, Atif and Sametinger, Johannes and Larrucea, Xabier and Fensel, Anna and Martinez-Gil, Jorge and Moser, Bernhard and Seifert, Christin and Stein, Benno and Granitzer, Michael",
    title="The Use of Class Assertions and Hypernyms to Induce and Disambiguate Word Senses",
    booktitle="Database and Expert Systems Applications",
    year="2019",
    publisher="Springer International Publishing",
    pages="172--181",
    isbn="978-3-030-27684-3"
    }```
3. https://medium.com/@revenkoartem/label-unstructured-data-using-enterprise-knowledge-graphs-2-d84bda281270
4. TO BE ADDED LATER

#### Used Language Model

The language model is loaded in [./ptlm_wsid/target_context.py](./ptlm_wsid/target_context.py) in `load_model`. The current implementation support BERT and DistilBERT from [HuggingFace Transformers](https://github.com/huggingface/transformers) library. By default `distilbert-base-multilingual-cased` is used, to change to BERT you can define the environment variable `TRANSFORMER_MODEL = bert-base-multilingual-cased`. For further models see the [HF documentation](https://huggingface.co/models).

## Run as Docker

You can use the `Dockerfile` and run the scripts in the docker container.

1. `cd` to the root folder where the `Dockerfile` is located.
2. run `DOCKER_BUILDKIT=1 docker build -t ptlm_wsid .`
3. run `docker run -it ptlm_wsid` By default, it will execute `./scripts/wsid_example.py` script, change the last line in the `Dockerfile` to run a different script.

## Usage

Find examples of usage in [./scripts](./scripts) folder.

### Installing

Easy install with pip from the github repo:
```bash
pip install git+https://github.com/semantic-web-company/ptlm_wsid.git#egg=ptlm_wsid
```

### License

This project is licensed under the MIT License.
