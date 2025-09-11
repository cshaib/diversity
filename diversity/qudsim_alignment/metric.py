import numpy as np

class SimilarityMetric():
    def _calculate_document_similarity(self, segment_scores):
        max_sim_scores = np.max(segment_scores, axis=1)
        overall_similarity = np.average(max_sim_scores)
        return  overall_similarity

class FrequencyBasedSimilarity(SimilarityMetric):
    
    def _count_sentences(self, qud_answers, 
                        num_target_segments:int, 
                        target_segments):
        
        arr = np.zeros((len(qud_answers['excerpts']), num_target_segments))

        for i, qud_ans in enumerate(qud_answers['excerpts']):
            answer_sentences = qud_ans['sentence_nums']
            if len(answer_sentences)==0:
                continue
            for j, target_segment in enumerate(eval(target_segments)['segmentation']):
                target_segment_sentences = set(target_segment['sentences'])
                intersection = target_segment_sentences.intersection(answer_sentences)
                arr[i][j] = len(intersection) / len(answer_sentences)

        return arr

    def _get_segment_scores(self, sentence_count_map, 
                            source_seg_qud_dict, 
                            num_source_segments, 
                            num_target_segments):
        
        segment_scores = np.zeros((num_source_segments, num_target_segments))

        for src, quds in source_seg_qud_dict.items():
            segment_scores[int(src)] = np.average(sentence_count_map[quds], axis=0)

        return segment_scores

    def calculate_similarity(self, num_target_segments:int, 
                             num_source_segments:int, 
                             qud_answers:list, 
                             source_seg_qud_dict: dict, 
                             target_segments: str): 
        
        sentence_count_map = self._count_sentences(qud_answers,
                                                   num_target_segments,
                                                   target_segments)
        
        segment_scores = self._get_segment_scores(sentence_count_map, 
                                                  source_seg_qud_dict, 
                                                  num_source_segments, 
                                                  num_target_segments)

        return segment_scores