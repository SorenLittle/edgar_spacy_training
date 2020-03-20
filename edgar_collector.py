import csv
import pandas as pd

from pprint import pprint
from edgar import (
    Company,
    Edgar,
)


def main():
    # initialize the EDGAR database
    edgar = Edgar()

    # has_ex_10(edgar)

    training_sentences = get_training_sentences('training_sentences/sentences_200_2.csv')

    entity_finder_ui(training_sentences)


# takes edgar, returns a cleaned pandas dataframe with names and cik's
def init_edgar_df(edgar):
    # move edgar data to pandas dataframe and clean the dataframe
    df = pd.DataFrame.from_dict(edgar.all_companies_dict, orient='index')
    df = df.reset_index().rename(columns={'index': 'name', 0: 'cik'})
    return df


def get_training_sentences(file):
    sentences = []
    with open(file, 'r') as f:
        reader = csv.reader(f)
        [sentences.extend(row) for row in reader]
    return sentences


# takes edgar, returns a database with all the company's who have exhibit 10's
def has_ex_10(edgar):
    company_df = init_edgar_df(edgar)
    company_df['ex-10'] = False

    for _, row in company_df.drop_duplicates(subset='cik', keep='first').iloc[4000:7000].iterrows():
        cik = row['cik']

        # initialize a Company instance
        company = Company(name=edgar.get_company_name_by_cik(cik), cik=cik)

        # get all the "EX-10" type documents from the company's 10K
        documents = company.get_document_type_from_10K('EX-10', no_of_documents=1)

        if documents:
            company_df.at[_, 'ex-10'] = True

    ex_10_df = company_df[company_df['ex-10'] == True]
    ex_10_df.to_csv('/Users/sorenlittle/PycharmProjects/edgar_spacy_training/ex_10_df/ex_10_df_4000_7000.csv')


def entity_finder_ui(sentences):
    print("-------------------------------------------------------------------------------------")
    print("                    FIND THE ENTITIES GAME copyright Soren Little")
    print("-------------------------------------------------------------------------------------")

    commands_string = '''
 - to mark an entity enter the entities full name (not case sensitive), hit enter, and then its spacy tag
 - when all entities have been marked hit enter to go to next sentence
 - to mark a sentence as useless type "d" and then enter
 - type 'exit game' to pause labelling
NOTE: some sentences should have NO entity in them, enter will mark it as no entities, "d" will delete it
    '''
    remaining_sentences = sentences[:]
    marked_up_sentences = []
    counter = 1

    for sentence in sentences:
        print(commands_string)
        print(f"PROGRESS: {counter} of {len(sentences)} sentences, only {len(sentences) - counter} to go!")

        pprint(f"\n{sentence}\n")

        entity_dict = {}

        entity = 'placeholder'
        while entity:

            entity = ''
            entity = input('Entity: ')
            if entity == 'd':
                print("sentence dropped...")
                entity_dict.update({entity: None})
                break
            if entity == 'exit game':
                print("exiting...")
                break
            if entity and entity != 'd':
                role = input('Role: ')
                entity_dict.update({entity: role})

        if entity == 'exit game':
            print('got there')
            with open('/Users/sorenlittle/PycharmProjects/edgar_spacy_training/temp_tagging_files/remaining_sentences.csv',
                      'w', newline='') as f:
                print('writing')
                wr = csv.writer(f, quoting=csv.QUOTE_ALL)
                wr.writerow(remaining_sentences)
            with open('/Users/sorenlittle/PycharmProjects/edgar_spacy_training/temp_tagging_files/marked_up_sentences.csv',
                      'w', newline='') as f:
                wr = csv.writer(f, quoting=csv.QUOTE_ALL)
                wr.writerow(marked_up_sentences)
            break

        if entity_dict == {}:
            marked_up_sentences.append((sentence, entity_dict))
        elif list(entity_dict.keys())[0] != 'd':
            marked_up_sentences.append((sentence, entity_dict))

        remaining_sentences.remove(sentence)
        if len(remaining_sentences) < 1:
            with open('/Users/sorenlittle/PycharmProjects/edgar_spacy_training/temp_tagging_files/marked_up_sentences.csv',
                      'w', newline='') as f:
                wr = csv.writer(f, quoting=csv.QUOTE_ALL)
                wr.writerow(marked_up_sentences)

        counter += 1


if __name__ == '__main__':
    main()
