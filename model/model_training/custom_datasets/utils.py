# mostly taken from
# https://huggingface.co/datasets/gozfarb/ShareGPT_Vicuna_unfiltered/blob/main/optional_clean.py,
# https://huggingface.co/datasets/ehartford/WizardLM_alpaca_evol_instruct_70k_unfiltered/blob/main/wizardlm_clean.py
FILTER_BY_WORDS = [
    "as a language model",
    "as an AI language model",
    "As a large language model",
    "As an AI ",
    "an AI language model you don't have",
    "As an AI language model, I cannot",
    "As an AI language model, I do not",
    "As an AI language model, I am not able",
    "As an AI language model, I don't have personal",
    "I am an AI language model and do not",
    "As an AI language model, I don't have",
    "As an AI language model, I am only able",
    "AI language model and I do not",
    "As an AI language model, I cannot modify",
    "As an AI language model, I do not",
    "I know as an AI language model you don't have",
    "as an AI language model, you cannot",
    "I'm sorry, but as an AI language model",
    "As an AI language model, I don't have",
    "I'm an AI ",
    "I am an AI ",
    "As your dedicated AI language model",
    "As a hypothetical AI",
    "As a neutral AI",
    "my knowledge cutoff",
    "my knowledge cut off",
    "As a machine",
    "I cannot assist",
    "I do not have personal preferences",
    "I don't have personal preferences",
    "Unfortunately, I cannot provide",
    "I'm sorry, I cannot",
    "I'm sorry, I cannot generate",
    "AI cannot create or program",
    "I'm afraid I cannot create",
    "OpenAI",
    "MOSS",
    "moss",
    "作为语言模型",
    "作为AI语言模型",
    "作为大型语言模型",
    "作为AI",
    "你没有的AI语言模型",
    "作为AI语言模型，我不能",
    "作为AI语言模型，我不会",
    "作为AI语言模型，我无法",
    "作为AI语言模型，我没有个人",
    "我是一个AI语言模型，不会",
    "作为AI语言模型，我没有",
    "作为AI语言模型，我只能",
    "AI语言模型，我不会",
    "作为AI语言模型，我不能修改",
    "作为AI语言模型，我不会",
    "我知道作为AI语言模型，你没有",
    "作为AI语言模型，你不能",
    "很抱歉，但作为AI语言模型",
    "作为AI语言模型，我没有",
    "我是一个AI",
    "我是一个AI",
    "作为你专属的AI语言模型",
    "作为一个假想的AI",
    "作为一个中立的AI",
    "我的知识截止日期",
    "我的知识切断",
    "作为一台机器",
    "我无法协助",
    "我没有个人偏好",
    "我没有个人偏好",
    "很抱歉，我无法提供",
    "很抱歉，我无法",
    "很抱歉，我无法生成",
    "AI无法创建或编程",
    "恐怕我无法创建",
]


def _filter_by_words(text: str, filter_words: list[str] | None = None) -> None | str:
    """Used to filter text that contains one of the `FILTER_BY_WORDS`. If so we return `None`
       otherwise we return the string

    Args:
        text (str): text to be filtered

    Returns:
        None | str: filtered text
    """
    filter_words = filter_words or FILTER_BY_WORDS
    for word in filter_words:
        if word.lower() in text.lower():
            return None
    return text
