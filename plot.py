import pickle
import matplotlib.pyplot as ppl

f = open("./history.pickle", "rb")

hist = pickle.load(f)

ppl.figure()

ppl.plot(range(1,51), hist['categorical_accuracy'])
ppl.plot(range(1,51), hist['val_categorical_accuracy'])
ppl.legend(("Trening", "Test"))
ppl.xlabel("Epoka")
ppl.ylabel("Dokładność")
ppl.grid(True)

ppl.savefig("prez/ml.png")
