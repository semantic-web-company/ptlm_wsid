# Example:
```
import target_context as tc
import cluster

top_n = 100
predicted = dict()
for sents, target_label in corpus:
    cxt = tc.SubstituteTargetContext(sents, target_label)
    top_pred, top_logit_scores = cxt.get_topn_predictions(top_n=top_n)
    predicted[doc.name] = top_pred
clusters = cluster.fca_cluster(predicted, target_label)
```

# TODOs:

- [ ] Implement the function to decide if category is relevant to `broader="cocktail"` or not in `main`
- [ ] Implement disambiguation: get substitutes and count how many belong to each induced cluster. normalize over cluster size.
- [ ] Implement disambiguation without clusters: provide sense indicators and check its probability. take maximum and compare to the most probably substitute
- [ ] Benchmark on cocktails