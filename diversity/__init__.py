from .compression import compression_ratio
from .patterns.token import token_patterns
from .patterns.part_of_speech import pos_patterns, get_pos
from .utils.memoize import memoized
from .homogenization import homogenization_score
from .ngram_diversity import ngram_diversity_score
from .functions import extract_patterns, match_patterns
from .self_repetition import self_repetition_score
from .embedding import remote_clique, chamfer_dist
from .template import template_rate, templates_per_token
from .compute_all_metrics import compute_all_metrics