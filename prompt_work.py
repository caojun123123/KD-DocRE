from openprompt.data_utils import InputExample
from openprompt.plms import load_plm
from openprompt.prompts import ManualTemplate
from openprompt.prompts import ManualVerbalizer
from openprompt import PromptForClassification
from openprompt import PromptDataLoader
import torch
import os
os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"

def century_prompt(dataset):
    classes = [
        "18",
        "19",
        "20"
    ]
    plm, tokenizer, model_config, WrapperClass = load_plm("bert", "bert-base-cased")
    promptVerbalizer = ManualVerbalizer(
        classes = classes,
        label_words = {
            "18": ["18th century", "1800s"],
            "19": ["19th century", "1900s"],
            "20": ["20th century", "2000s"]
        },
        tokenizer = tokenizer,
    )
    plm, tokenizer, model_config, WrapperClass = load_plm("bert", "bert-base-cased")
    promptTemplate = ManualTemplate(
        text = '{"placeholder":"text_a"} , As we can see from the above , {"placeholder":"text_b"} in the {"mask"}',
        tokenizer = tokenizer,
    )

    promptModel = PromptForClassification(
        template = promptTemplate,
        plm = plm,
        verbalizer = promptVerbalizer,
    )

    data_loader = PromptDataLoader(
        dataset = dataset,
        tokenizer = tokenizer,
        template = promptTemplate,
        tokenizer_wrapper_class=WrapperClass,
    )


    # making zero-shot inference using pretrained MLM with prompt
    promptModel.eval()
    with torch.no_grad():
        for batch in data_loader:
            logits = promptModel(batch)
            preds = torch.argmax(logits, dim = -1)
            print("century" + str(classes[preds]))

def age_prompt(dataset):
    classes = [ 
        "0","1","2","3","4","5","6","7","8","9"
    ]
    plm, tokenizer, model_config, WrapperClass = load_plm("bert", "bert-base-cased")
    promptVerbalizer = ManualVerbalizer(
        classes = classes,
        label_words = {
            "0": ["00s", "aughts"],
            "1": ["10s", "tens"],
            "2": ["20s", "twenties"],
            "3": ["30s", "thirties"],
            "4": ["40s", "forties"],
            "5": ["50s", "fifties"],
            "6": ["60s", "sixties"],
            "7": ["70s", "seventies"],
            "8": ["80s", "eighties"],
            "9": ["90s", "nineties"],
        },
        tokenizer = tokenizer,
    )
    plm, tokenizer, model_config, WrapperClass = load_plm("bert", "bert-base-cased")
    promptTemplate = ManualTemplate(
        text = '{"placeholder":"text_a"} , As we can see from the above , {"placeholder":"text_b"} in the {"mask"}',
        tokenizer = tokenizer,
    )

    promptModel = PromptForClassification(
        template = promptTemplate,
        plm = plm,
        verbalizer = promptVerbalizer,
    )

    data_loader = PromptDataLoader(
        dataset = dataset,
        tokenizer = tokenizer,
        template = promptTemplate,
        tokenizer_wrapper_class=WrapperClass,
    )


    # making zero-shot inference using pretrained MLM with prompt
    promptModel.eval()
    with torch.no_grad():
        for batch in data_loader:
            logits = promptModel(batch)
            preds = torch.argmax(logits, dim = -1)
            print("age" + str(classes[preds]))
    # predictions would be 1, 0 for classes 'positive', 'negative'


dataset = [ # For simplicity, there's only two examples
    # text_a is the input text of the data, some other datasets may have multiple input sentences in one example.
    InputExample(
        guid = 0,
        text_a = "Obama served as President of the United States at 2009",
        text_b = "Obama served as President of the United States"
    ),
    InputExample(
        guid = 0,
        text_a = "Obama served as President of the United States at 1980",
        text_b = "Obama served as President of the United States"
    )
]

century_prompt(dataset)
age_prompt(dataset)



 