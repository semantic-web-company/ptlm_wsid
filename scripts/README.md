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

To run use
`python3 wikiner.py configs/wikiner_en_config.conf` -- to run with English WikiNer.

