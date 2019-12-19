# Pretrained Language Model for WSID

In induction step the service takes a list of context and target in them (specified by a list of `(start_index, end_index)` tuples).
The service uses pretrained language model ([BERT](https://github.com/google-research/bert)) to get predictions of possible substitutes for the target word. Next the service clusters the obtained substitutes using [FCA](https://www.upriss.org.uk/fca/fca.html). 
Namely, we use Algorithm 2 from [https://doi.org/10.1016/j.jcss.2009.05.002] to obtain the binary factors and filter the clusters to obtain the senses.

# Deployment

do 

```RUN python -m nltk.downloader punkt stopwords averaged_perceptron_tagger wordnet -d /wsid_web_service/nltk```

to install the necessary nltk data.

## TODO
