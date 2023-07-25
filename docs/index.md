# Inid

**Inid** (**In**itialize **id**entification) is a library to kickstart your next PyTorch-based recognition project.

In contrast with a classification model, a recognition model should be able to cope with classes that were not present in the training set. Therefore, instead of outputting a class, a recognizer outputs an *embedding*. By comparing the embedding of a given *query* image with the embeddings of labeled references (the *gallery*), we can predict a label for the query.

```{toctree}
:caption: User guide
:maxdepth: 2

guide/01-getting_started
guide/02-customize
guide/03-kfold_cross_val
```

```{toctree}
:caption: API
:maxdepth: 4

data <api/data>
eval <api/eval>
model <api/model>
utils <api/utils>
```

# Indices and tables

- {ref}`genindex`
- {ref}`search`
