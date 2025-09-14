from .qudsim_modules import number_text, get_quds, align
from .utils import openai
from tqdm import tqdm
import itertools
import json
import yaml
import os

class Document:
    def __init__(self, document):
        self.document = document

    def preprocess_document(self):

        numbered_text, number_sentence_dict = number_text(self.document)

        if not numbered_text or not number_sentence_dict:
            print("Could not preprocess document: ", self.document)
            self.numbered_text = None
            self.number_sentence_dict = None
            return False
        
        self.numbered_text = numbered_text
        self.number_sentence_dict = number_sentence_dict
        return True

    def generate_quds(self, gpt_model, config):
        try:
            level = config['level']
            max_tries = config['max_tries']
        except:
            print("Failed to parse configurations")
            return None, None, [], []
        
        qg_item = get_quds(gpt_model, self.numbered_text, self.number_sentence_dict, level, max_tries)
        
        if qg_item is None:
            print("Could not segment, abstract or generate quds")
            self.segments = None
            self.entity_abstracted_segments = None
            self.quds = None
            self.segment_qud_dict = None
            self.qud_segment_dict = None
            return False
        else:
            self.segments = qg_item['segments']
            self.entity_abstracted_segments = qg_item['entity_abstracted_segments']
            self.quds = qg_item['quds']
            self.segment_qud_dict = qg_item['segment_qud_dict']
            self.qud_segment_dict = qg_item['qud_segment_dict']
            return True
    
class AlignmentPair:
    def __init__(self, document1: Document, document2: Document):
        self.source_document = document1
        self.target_document = document2

    def align_documents(self, gpt_model, config):
        try:
            threshold = config['threshold']
            max_tries = config['max_tries']
        except:
            print("Failed to parse configurations")
            return None, None, [], []
        
        num_source_segments = len(self.source_document.segment_qud_dict)
        num_target_segments = len(self.target_document.segment_qud_dict)
        num_source_sentences = len(self.source_document.number_sentence_dict)
        num_target_sentences = len(self.target_document.number_sentence_dict)
        source_segments = self.source_document.segments
        target_segments = self.target_document.segments
        
        source_qud_answers, target_qud_answers, harmonic_mean_scores, aligned_segments = align(gpt_model,
                                                                                                     self.source_document.quds,
                                                                                                     self.target_document.numbered_text,
                                                                                                     self.target_document.quds,
                                                                                                     self.source_document.numbered_text,
                                                                                                     num_source_segments,
                                                                                                     num_target_segments,
                                                                                                     self.source_document.segment_qud_dict,
                                                                                                     self.target_document.segment_qud_dict,
                                                                                                     self.source_document.segments,
                                                                                                     self.target_document.segments,
                                                                                                     num_source_sentences,
                                                                                                     num_target_sentences,
                                                                                                     threshold,
                                                                                                     max_tries)

        self.source_qud_answers = source_qud_answers
        self.target_qud_answers = target_qud_answers
        self.harmonic_mean_scores = harmonic_mean_scores.tolist()
        self.aligned_segments = aligned_segments.tolist()

        aligned_segment_text = []
        for i, src in enumerate(aligned_segments):
            for j, tgt in enumerate(src):
                if tgt>0:
                    # alignment exists
                    source_sentences = eval(source_segments)['segmentation'][i]['sentences']
                    try:
                        source_text = [self.source_document.number_sentence_dict[str(num)] for num in source_sentences]
                    except:
                        try:
                            source_text = [self.source_document.number_sentence_dict[num] for num in source_sentences]
                        except Exception as e:
                            print(e)

                    target_sentences = eval(target_segments)['segmentation'][j]['sentences']
                    try:
                        target_text = [self.target_document.number_sentence_dict[str(num)] for num in target_sentences]
                    except:
                        try:
                            target_text = [self.target_document.number_sentence_dict[num] for num in target_sentences]
                        except Exception as e:
                            print(e)
                    
                    aligned_segment_text.append((" ".join(source_text), " ".join(target_text)))
        
        self.aligned_segment_text = aligned_segment_text


def _compile_documents(documents: list[str], qg_gpt_model: openai.GPT, config):
    document_list = []
    for document in tqdm(documents, total=len(documents), desc="Generating QUDs"):
        document_obj = Document(document=document)

        preprocessing_status = document_obj.preprocess_document()
        if not preprocessing_status:
            continue

        qud_generation_status = document_obj.generate_quds(gpt_model=qg_gpt_model, config=config)
        if not qud_generation_status:
            continue

        document_list.append(document_obj)

    return document_list

def _custom_serializer(obj):
    if hasattr(obj, '__dict__'):
        return obj.__dict__
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def qudsim(documents: list[str], key=None, config_file=None):
    """
    Args:
        documents (list[str]): a list of texts to be aligned (all combinations of pairs will be computed)
        key (str): OpenAI Key
        config_file (str): a .yaml or .yml file that contains the necessary configurations (see config.yaml for the default config)

    Returns:

        
    """

    if not config_file:
        config_file = os.path.join(os.path.dirname(__file__), '../config.yaml')

    with open(config_file, 'r') as file:
        configs = yaml.safe_load(file)

    try:
        qg_gpt_model_name = configs['qg_gpt_model']
        qa_gpt_model_name = configs['qa_gpt_model']
        level = configs['level']
        threshold = configs['threshold']
        max_tries = configs['max_tries']
    except:
        print("Failed to parse configurations")
        return


    
    qg_gpt_model = openai.GPT(qg_gpt_model_name, key=key)
    qa_gpt_model = openai.GPT(qa_gpt_model_name, key=key)
    
    # level of abstraction of QUDs, with 0 being highly specific and 1 being abstractive
    
    if level!=0 and level!=1:
        print("Levels 0 and 1 are supported, 0 being specific and 1 being abstract. Value passed was an unsupported level.")
        return
    
    if threshold < 0 or threshold > 1:
        print("Threshold value is outside the valid range (0,1).")
        return
    
    if max_tries<=0:
        print("Maximum number of attempts to successfully align pairs must be at least 1.")
        return
    
    if len(documents)<2:
        print("At least two documents must be provided.")
        return

    # create document objects (one Document per document)
    document_list = _compile_documents(documents=documents, qg_gpt_model=qg_gpt_model, config=configs)

    if len(document_list)<2:
        print("At least two documents must successfully generate QUDs.")
        return

    # make pairs from the list of Document objects
    pair_combinations = list(itertools.combinations(document_list, 2))
    alignment_pairs = []
    for doc1, doc2 in tqdm(pair_combinations, total=len(pair_combinations), desc='Aligning Document Pairs'):
        alignment_pair = AlignmentPair(document1=doc1, document2=doc2)
        alignment_pair.align_documents(gpt_model=qa_gpt_model, config=configs)
        alignment_pairs.append(alignment_pair)

    
    json_str = json.dumps(alignment_pairs, default=_custom_serializer)
    return json_str
