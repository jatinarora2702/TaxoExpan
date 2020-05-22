import tensorflow_hub as hub


def embed_taxo_terms(model, termsfile, outfile):
    print("loading taxo terms")
    ids = []
    terms = []
    with open(termsfile, "r") as f:
        for line in f:
            s = line.split("\t")
            ids.append(s[0].strip())
            terms.append(s[1].lower().strip())

    print("calculating embeddings")
    emb = model(terms)

    print("saving embeddings")
    with open(outfile, "w") as f:
        f.write(str(len(terms)) + " " + str(len(emb[0].numpy())) + "\n")
        for i, v in enumerate(emb):
            f.write(ids[i] + " " + " ".join([str(t) for t in list(v.numpy())]) + "\n")
    print("done")


if __name__ == '__main__':
    print("loading model")
    model = hub.load("/shared/data/jatin2/universal-sentence-encoder/model/")

    # embed_taxo_terms(model, "../data/MAG_FoS/computer_science.terms", "../data/MAG_FoS/computer_science.terms.univ.embed")
    embed_taxo_terms(model, "../data/A2/a2.terms", "../data/A2/a2.terms.univ.embed")
