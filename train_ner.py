from __future__ import unicode_literals, print_function

# import plac
import random
from pathlib import Path
import spacy
from spacy import displacy
from spacy.util import minibatch, compounding

# training data
TRAIN_DATA = []


# plac annotations allow the function to be called from the terminal through "python train_ner.py en_core_web_sm /path"
# not necessary for calling within python
# @plac.annotations(
#     model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
#     new_model_name=("New model name for model meta.", "option", "nm", str),
#     output_dir=("Optional output directory", "option", "o", Path),
#     n_iter=("Number of training iterations", "option", "n", int),
# )
def main(model=None, new_model_name='legal_ent', output_dir=None, n_iter=100):
    # load training training data
    # TODO: add real training data, import the training_sentences at the top
    for sentence in training_sentences:
        TRAIN_DATA.append(train_data_strings(sentence[0], sentence[1]))

    # load the model, set up the pipeline, and train the entity recognizer
    random.seed(0)
    if model is not None:
        nlp = spacy.load(model)  # load existing spacy model
        print(f"Loaded model {model}")
    else:
        nlp = spacy.blank("en")  # create a blank language class
        print("Created blank 'en' model")

    # create the built-in pipline components and add them to the pipeline
    # if nlp.create_pipe works for spacy built-ins
    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner, last=True)
    # otherwise, get it so we can add labels
    else:
        ner = nlp.get_pipe("ner")

    # add labels
    for _, annotations in TRAIN_DATA:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])
    if model is None:
        optimizer = nlp.begin_training()
    else:
        optimizer = nlp.resume_training()
    move_names = list(ner.move_names)

    # get names of other pipes to disable them during training
    pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
    with nlp.disable_pipes(*other_pipes):  # only train NER
        # batch up the examples using minibatch
        for itn in range(n_iter):
            random.shuffle(TRAIN_DATA)
            batches = minibatch(TRAIN_DATA, size=compounding(1.0, 4.0, 1.001))
            losses = {}
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, sgd=optimizer, drop=0.35, losses=losses)
            print("Losses", losses)

    # test the trained model
    test_text = 'This non-disclosure agreement (“Agreement”) between Soren Company LLC doing business as SorenComp and Berkman LLC doing business as Berkman Solutions.'

    doc = nlp(test_text)
    print(f"Entities in {test_text}")
    for ent in doc.ents:
        print(ent.label_, ent.text)

    # for text, _ in TRAIN_DATA:
    #     doc = nlp(text)
    #     print("Entities", [(ent.text, ent.label_) for ent in doc.ents])
    #     print("Tokens", [(t.text, t.ent_type_, t.end_iob) for t in doc])

    # save the model to output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.meta["name"] = new_model_name  # rename model
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

        # test the saved model
        print("Loading from", output_dir)
        nlp2 = spacy.load(output_dir)
        # check the classes have been loaded back consistently
        assert nlp2.get_pipe("ner").move_names == move_names
        doc2 = nlp2(test_text)
        for ent in doc2.ents:
            print(ent.label_, ent.text)
            displacy.serve(doc2, style='ent')


def train_data_strings(input_string, entities_dict):
    entities = []
    i = 0
    for name, label in entities_dict.items():
        start = input_string.lower().find(name.lower(), i)
        end = start + len(name)

        i += end

        entities.append((start, end, label))
    return input_string, {"entities": entities}


if __name__ == '__main__':
    main(output_dir="//models")

    # use below instead if running in terminal, use with annotations above main()
    # plac.call(main(output_dir="/Users/sorenlittle/PycharmProjects/edgar_spacy_training/models"))
