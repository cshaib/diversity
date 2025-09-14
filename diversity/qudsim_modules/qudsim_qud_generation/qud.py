from ...utils import openai

system_prompt_high_level = "You will be given a paragraph. We are interested in forming unique, high-level, abstract QUDs with minimal details such that when they are answered, we understand the main themes of the paragraph. Details specific to the content should be omitted. QUDs should like: What were the individual's greatest accomplishments? What legacy did the individual leave behind?. First answer the minimum number of QUD(s) required. Then list the QUDs. Do not use conjunctions in the QUDs."

class QUD(openai.BaseModel):
    qud: str

class Answer(openai.BaseModel):
    num_quds: int
    quds: list[QUD]

def generate_quds(gpt_model: openai.GPT, segment: str, max_tries: int):
    """Generate Question(s) Under Discussion for a given segment

    Args:
        gpt_model: an OpenAI client
        segment (str): the segment for which QUDs need to be generated (QUDs are answered by the segment)
        max_tries (int): maximum number of attempts the client can make in case of failure 
    
    Returns:
        Answer: QUDs wrapped in an openai.BaseModel class
    """
    for i in range(max_tries):
        try:
            quds = gpt_model.call_gpt_format(segment, system_prompt_high_level, Answer)

            if not quds or len(quds.quds)==0:
                continue

            return quds
        except Exception as e:
            continue
    
    return None
            