from .answer import get_answer
from .similarity import get_harmonic_similarity
import numpy as np

def align(gpt_model,
          source_quds, 
          target_text, 
          target_quds, 
          source_text,
          num_source_segments,
          num_target_segments,
          source_segment_qud_dict,
          target_segment_qud_dict,
          source_segments,
          target_segments,
          num_source_sentences,
          num_target_sentences,
          threshold,
          max_tries):
    
    # get [A_q for q in source_quds]
    quds = []
    for q_group in source_quds:
        for q in eval(q_group)['quds']:
            quds.append(q['qud'])

    source_qud_list = "\n\n".join(quds)
    source_qud_answers = get_answer(gpt_model, target_text, source_qud_list, len(quds), num_target_sentences, max_tries)
    
    if source_qud_answers is None:
        print("Finding an answer given source QUDs and target document was unsuccessful")
        return None, None, [], []
    else:
        source_qud_answers = eval(source_qud_answers)
    
    # get [A_q for q in target_quds]
    quds = []
    for q_group in target_quds:
        for q in eval(q_group)['quds']:
            quds.append(q['qud'])

    target_qud_list = "\n\n".join(quds)
    target_qud_answers = get_answer(gpt_model, source_text, target_qud_list, len(quds), num_source_sentences, max_tries)
    
    if target_qud_answers is None:
        print("Finding an answer given target QUDs and source document was unsuccessful")
        return None, None, [], []
    else:
        target_qud_answers = eval(target_qud_answers)

    # get harmonic similarity
    harmonic_mean_scores = get_harmonic_similarity(num_target_segments, 
                                                    num_source_segments, 
                                                    source_qud_answers,
                                                    source_segment_qud_dict,
                                                    target_segments,
                                                    target_qud_answers,
                                                    target_segment_qud_dict,
                                                    source_segments)
    
    aligned_segments = np.where(np.array(harmonic_mean_scores)<threshold, 0, 1)

    return source_qud_answers, target_qud_answers, harmonic_mean_scores, aligned_segments