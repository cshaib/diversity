from ...utils import openai

system_prompt = "You will be given several numbered paragraphs. Decontextualize each paragraph such that the paragraph's general plot is captured. Names, places, extraneous details and descriptive language should all be abstracted away."

class Paragraph(openai.BaseModel):
    para_num: int
    para: str

class Answer(openai.BaseModel):
    decontextualized_paragraphs: list[Paragraph]

def decontextualize(gpt_model: openai.GPT, text: str, num_segments: int, max_tries: int):
    """Extracts and removes entities from segments

    Args:
        gpt_model: an OpenAI client
        text (str): string concatenation of segments (default delimeter is two newlines) 
        num_segments (int): number of segments being entity-extracted
        max_tries (int): maximum number of attempts the client can make in case of failure 
    
    Returns:
        Answer: decontextualized segments wrapped in an openai.BaseModel class
    """

    for i in range(max_tries):
        try:
            decontextualized = gpt_model.call_gpt_format(text, system_prompt, Answer)

            if not decontextualized:
                continue

            if len(decontextualized.decontextualized_paragraphs)!=num_segments:
                continue

            return decontextualized
        except Exception as e:
            continue

    return None

    
