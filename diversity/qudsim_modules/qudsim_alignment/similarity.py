from .metric import FrequencyBasedSimilarity
import numpy as np

def _get_frequency_similarities(num_target_segments, 
                               num_source_segments, 
                               qud_answers, 
                               source_segment_qud_dict,
                               target_segments):
    similarity_metric = FrequencyBasedSimilarity()

    segment_scores = similarity_metric.calculate_similarity(num_target_segments, 
                                                            num_source_segments, 
                                                            qud_answers,
                                                            source_segment_qud_dict,
                                                            target_segments)
    return segment_scores

def get_harmonic_similarity(num_target_segments,
                            num_source_segments,
                            source_qud_answers,
                            source_segment_qud_dict,
                            target_segments,
                            target_qud_answers,
                            target_segment_qud_dict,
                            source_segments):
    """ Calculates harmonic mean between source->target and target->source similarities

        Args:
            num_target_segments (int): Number of target segments
            num_source_segments (int): Number of source segments
            source_qud_answers (dict): JSON representation of answers to source quds given target document
            source_seg_qud_dict (dict): Mapping between source segment indices and a list of corresponding QUD indices
            target_segments (str): JSON string representation of target segments (qudsim_qud_generation.segment.Answer)
            target_qud_answers (dict): JSON representation of answers to target quds given source document
            target_seg_qud_dict (dict): Mapping between target segment indices and a list of corresponding QUD indices
            source_segments (str): JSON string representation of source segments (qudsim_qud_generation.segment.Answer)
        
        Returns:
            ndarray: array of dimensions (num_source_segments, num_target_segments)
                representing harmonic mean of direction similarity scores between each pair of segments
        """
    
    src_to_tgt_segment_scores = _get_frequency_similarities(num_target_segments,
                                                            num_source_segments,
                                                            source_qud_answers,
                                                            source_segment_qud_dict,
                                                            target_segments)
    tgt_to_src_segment_scores = _get_frequency_similarities(num_source_segments,
                                                            num_target_segments,
                                                            target_qud_answers,
                                                            target_segment_qud_dict,
                                                            source_segments)

    denom = src_to_tgt_segment_scores + np.transpose(tgt_to_src_segment_scores)
    denom = np.where(denom>0, denom, 1)

    harmonic_mean_scores = 2*(src_to_tgt_segment_scores*np.transpose(tgt_to_src_segment_scores))/denom

    return harmonic_mean_scores