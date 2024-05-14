run_easg.py is the original file from the paper implementation

edge_cls is just a console to run some experiments, can be ignored


the dataset had a bug: when adding 198 to object to make indicization unique for edge index, it overwrites the original dataset if we don't perform deepcopy. As a result the evaluation was incorrect because the triplets (only those with one object) had the wrong label
