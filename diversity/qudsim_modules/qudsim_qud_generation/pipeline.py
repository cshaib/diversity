from .decontextualize import decontextualize
from .qud import generate_quds
from .segment import segment

def _get_qud_dict(quds):
    qud_segment_dict = {}
    segment_qud_dict = {}
    num_quds = 0

    for i, source_qud in enumerate(quds):
        qud_idx_list = []
        for q in eval(source_qud)['quds']:
            qud_segment_dict[num_quds] = i
            qud_idx_list.append(num_quds)
            num_quds+=1
        segment_qud_dict[i] = qud_idx_list
            
    return segment_qud_dict, qud_segment_dict

def get_quds(gpt_model, text: str, sentence_num_dict: dict, level: int, max_tries: int):
    """Performs segmentation, entity extraction and QUD generation on the text at the specified level

    Args:
        gpt_model: an OpenAI client
        text (str): passage/document
        sentence_dict (dict): Dictionary mapping between sentence numbers and corresponding sentences in the passage
        level (0/1): Abstraction level of QUDs (1=abstract; 0=specific)
        max_tries (int): maximum number of attempts the client can make in case of failure 
    
    Returns:
        dict: results of the pipeline:
            segments: segmentation json
            segment_dict: Maps sentence number to its corresponding segment
            entity_abstracted_segments: segments post-entity-extraction
            quds: quds for the document in json
            segment_qud_dict: Maps segment index to a list of corresponding QUD indices
            qud_segment_dict: Maps QUD indices to their corresponding segment indices
    """

    # SEGMENTATION
    segments, segmented_text = segment(gpt_model, text, sentence_num_dict, max_tries)
    if segments is None or segmented_text is None:
        print("Segmentation was unsuccessful")
        return None
    
    segments_json = segments.model_dump_json()
    
    # DICTIONARY: SENTENCE --> SEGMENT
    segment_dict = {}
    for i, s in enumerate(segments.segmentation):
        for sentence_num in s.sentences:
            segment_dict[sentence_num] = i

    if level==1:
        # ENTITY ABSTRACTION
        numbered_segment_text = "\n\n".join(["[" + str(i)+ "] " + seg for i, seg in enumerate(segmented_text)])
        decontextualized_segments = decontextualize(gpt_model, 
                                                    numbered_segment_text, 
                                                    len(segmented_text), 
                                                    max_tries)
        if decontextualized_segments is None:
            print("Entity abstraction was unsuccessful.")
            return None
        
        decontextualized_segments_json = decontextualized_segments.model_dump_json()

        # QUD GENERATION - Level 1
        quds = [(generate_quds(gpt_model, seg.para, max_tries)).model_dump_json() 
                        for seg in decontextualized_segments.decontextualized_paragraphs]
        if quds is None:
            return None
    else:
        decontextualized_segments_json = None

        # QUD GENERATION - Level 0
        quds = [(generate_quds(gpt_model, seg, max_tries)).model_dump_json() for seg in segmented_text]
        if quds is None:
            return None

    segment_qud_dict, qud_segment_dict = _get_qud_dict(quds)
    qg_output_item = {"segments": segments_json,
                      "segment_dict": segment_dict,
                      "entity_abstracted_segments": decontextualized_segments_json,
                      "quds": quds,
                      "segment_qud_dict": segment_qud_dict,
                      "qud_segment_dict": qud_segment_dict}

    return qg_output_item 