import os
import time

import spacy
import json

from bs4 import BeautifulSoup
from transformers import BertTokenizer

#install scispacy & download 'en_core_sci_lg' model
NLP = spacy.load("en_core_sci_lg")
#using a word tokenizer (BertTokenizer) that is used for finetuning BERT models to the NER task
WORD_TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased')
ANNOTATORS = ['annotator1', 'annotator2','annotator3','annotator4','annotator5']
#Assign the directories of the unzip folders (annotation_plain_html_new.zip, annotation_json.zip) and annotation_legend.json file to HTML_PATH, JSON_PATH, ANNOTATION_LEGEND_DIC, respectively
HTML_PATH = 'corpus-for-DFI-extraction/dataset/annotation_plain_html_new'
JSON_PATH = 'corpus-for-DFI-extraction/dataset/annotation_json_new'
ANNOTATION_LEGEND_DIC = 'corpus-for-DFI-extraction/dataset/annotation_legend.json'

class Preprocessor():
    def __init__(self, html_path, json_path):
        self.html_path = html_path
        self.json_path = json_path
        self.html_files = [file for file in os.listdir(html_path) if "html" in file]
        self.json_files = [file for file in os.listdir(json_path) if "json" in file]
        
        with open(ANNOTATION_LEGEND_DIC, 'r', encoding='utf-8') as json_f:
            self.annotation_legend_dic = json.loads(json_f.read())
            self.annotation_legend_dic = dict((value, key) for key,value in self.annotation_legend_dic.items())
            self.annotation_legend_dic['food_component'] = ['e_64', 'e_61']
        
        annotator_list = map(lambda x: x.index(max(x)), [[html_file.find(annotator) for annotator in ANNOTATORS] for html_file in self.html_files])
        self.annotator_list = [ANNOTATORS[index] for index in annotator_list]

        #checking whether the orders of html_files and json_files are same
        for html_file, json_file in zip(self.html_files, self.json_files):
            if(html_file.replace('.plain.html', '')!=json_file.replace('.ann.json', '')):
                print('Check the orders of json files and html files.')
                print('Current html file is {}.'.format(html_file))
                print('Current json file is {}.'.format(json_file))
                break

        self._parsing_html_files()
        self._parsing_json_files()

    def _parsing_html_files(self):
        self.soups = []
        self.PMIDs = []; self.titles = []; self.main_texts = []; self.main_text_parts = []      

        for html_file in self.html_files:
            f = open(os.path.join(self.html_path, html_file), 'r', encoding='utf-8')
            html = f.read()
            soup = BeautifulSoup(html, 'html.parser')
            self.soups.append(soup)

            title, PMID, main_text, main_text_part = self._find_title_PMID_maintext_from_soup(soup)

            self.titles.append(title)
            self.PMIDs.append(PMID)
            self.main_texts.append(main_text)
            self.main_text_parts.append(main_text_part)

    def _parsing_json_files(self):
        self.json_data = []       
        self.metas = []; self.evdience_levels = []; self.inclusion_keys = []

        for json_file in self.json_files:
            with open(os.path.join(self.json_path, json_file), 'r', encoding='utf-8') as json_f:
                json_data_indi = json.load(json_f)

                meta_info, evidence_level, inclusion_key = self._find_meta_info_from_json(json_data_indi)

                self.json_data.append(json_data_indi)
                self.metas.append(meta_info)
                self.evdience_levels.append(evidence_level)
                self.inclusion_keys.append(inclusion_key)

    #parsing titles, PMIDs, and main_texts of abstracts from html files 
    def _find_title_PMID_maintext_from_soup(self, soup):        
        for html_section in  soup.find_all('section'):
            if(html_section.find('h2').contents[0].lower() in ['title', '\ufefftitle']):
                title_section = html_section
            elif(html_section.find('h2').contents[0].upper() in ['PMID', '\ufeffPMID']):
                PMID_section = html_section
            elif(html_section.find('h2').contents[0].lower() in ['main_text', '\ufeffmain_text', 'main text', '\ufeffmain text']):
                main_text_section = html_section
                main_text_part = main_text_section.find('p').get('id')

        title, PMID, main_text = map(lambda section: section.find('p').contents[0], [title_section, PMID_section, main_text_section])
        main_text = str(main_text).strip()

        return(title, PMID, main_text, main_text_part)

    #parsing meta-information of annotation from json files
    def _find_meta_info_from_json(self, json_data):
        meta_info = json_data['metas']
        try: 
            evidence_level = meta_info['m_24']['value']
        except:
            evidence_level = 'false'
        try:
            inclusion_key = meta_info['m_17']['value']
        except:
            inclusion_key = 'False'
        
        return(meta_info, evidence_level, inclusion_key)

    #preprocessing for sentence classification
    def preprocessing_for_sent_classification(self):
        print('Start preprocessing for sentence classification. \n Input-target tuple will be save in preprocessor.dataset_for_sent_classification')
        self.dataset_for_sent_classification = []
        start_time = time.time()

        for index, (main_text, json_data) in enumerate(zip(self.main_texts, self.json_data)):
            sents_in_main_text = [NLP(str(sent)) for sent in NLP(main_text).sents]
            sent_entities = [(NLP(str(entity['offsets'][0]['text'])), entity['classId']) for entity in json_data['entities'] if (entity['classId'] in ['e_27', 'e_30', 'e_34', 'e_50'])]

            sent_cls_targets = []
            for sent in sents_in_main_text:
                sent_similarities = [sent.similarity(sent_entity) for sent_entity, _ in sent_entities]

                try:
                    max_similarity = max(sent_similarities)
                    max_entity_type = sent_entities[sent_similarities.index(max_similarity)][1]
                except ValueError:
                    max_similarity = 0; max_entity_type = 'none'

                if(max_similarity>0.9):
                    sent_cls_targets.append((str(sent).strip("\'\" "), max_entity_type))
                else:
                    sent_cls_targets.append((str(sent).strip("\'\" "), 'none'))
            self.dataset_for_sent_classification.append(sent_cls_targets)

            if(index%500==0):
                print('* Current document index is %d.'%index)
                print('* Elapsed time is %d seconds.'%(time.time()-start_time))
                print('-'*80)

        print('Finished preprocessing for sentence classification! \n')

    #preprocessing for named entity recognition classification
    def preprocessing_for_NER(self):
        print('Start preprocessing for NER. \n Input-entity string-target tuple will be save in preprocessor.dataset_for_NER.')
        self.dataset_for_NER = []
        start_time = time.time()

        for index, (main_text, json_data, main_text_part) in enumerate(zip(self.main_texts, self.json_data, self.main_text_parts)):
            words_in_main_text = [word for word in NLP(main_text)]
            entities_in_main_text = [entity for entity in json_data['entities'] if (entity['part']==main_text_part)]
            word_entities_in_main_text = [(entity['offsets'][0]['start'], entity['offsets'][0]['text'],  entity['classId']) for entity in entities_in_main_text if (entity['classId'] in ['e_26', 'e_28', 'e_29', 'e_32', 'e_33', 'e_51', 'e_61'])]

            word_cls_targets = []
            for word in words_in_main_text:
                word_distances = [abs(word.idx - word_entity_index) for word_entity_index, _, _ in word_entities_in_main_text]

                try:
                    min_word_distance = min(word_distances)
                    min_entity, min_entity_type = word_entities_in_main_text[word_distances.index(min_word_distance)][1:]
                except ValueError:
                    min_word_distance = 100
                    min_entity, min_entity_type = '<WAS-NOT-WORD-ENTITY>', 'none'

                # print(word)
                # print(min_word_distance)
                # print(min_entity)
                # print(min_entity_type)
                # print('------------------------')

                if (min_word_distance<=5) & (str(word) in min_entity):
                    for token in WORD_TOKENIZER.tokenize(str(word)):
                        word_cls_targets.append((token, min_entity, min_entity_type))
                else:
                    for token in WORD_TOKENIZER.tokenize(str(word)):
                        word_cls_targets.append((token,'<WAS-NOT-WORD-ENTITY>', 'none'))

            self.dataset_for_NER.append(word_cls_targets)

            if(index%500==0):
                print('* Current document index is %d.'%index)
                print('* Elapsed time is %d seconds.'%(time.time()-start_time))
                print('-'*80)

        print('Finished preprocessing for NER! \n')


    #preprocessing for evidence-level classification / document classification
    def preprocessing_for_doc_classification(self):
        print('Start preprocessing for document classification. \n Input-entity string-target tuple will be save in preprocessor.dataset_for_doc_classification.')
        self.dataset_for_doc_classification = []
        start_time = time.time()

        for index, (main_text, evdience_level) in enumerate(zip(self.main_texts, self.evdience_levels)):
            self.dataset_for_doc_classification.append((main_text, evdience_level))

            if(index%500==0):
                print('* Current document index is %d.'%index)
                print('* Elapsed time is %d seconds.'%(time.time()-start_time))
                print('-'*80)

        print('Finished preprocessing for document classification! \n')

#Sample code
# preprocessor = Preprocessor(html_path=HTML_PATH, json_path=JSON_PATH)
# preprocessor.preprocessing_for_sent_classification()
# preprocessor.preprocessing_for_NER()
# preprocessor.preprocessing_for_doc_classification()