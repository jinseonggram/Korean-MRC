from transformers import AutoModelForQuestionAnswering, AutoTokenizer


def get_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer


def get_model(model_name):
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    return model