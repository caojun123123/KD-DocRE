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



classes = [ # There are two classes in Sentiment Analysis, one for negative and one for positive
    "2009",
    "2000"
]

dataset = [ # For simplicity, there's only two examples
    # text_a is the input text of the data, some other datasets may have multiple input sentences in one example.
    InputExample(
        guid = 0,
        text_a = "Obama served as President of the United States at 2009",
    ),
    InputExample(
        guid = 0,
        text_a = "Obama served as President of the United States at 2000",
    )
]

entity_subject = "Obama"
entity_object = "President of the United States"
predicate = "server"


prompt_text = '{"placeholder":"text_a"} , '+ entity_subject + ' ' + predicate + ' ' + entity_object + ' at {"mask"}'
plm, tokenizer, model_config, WrapperClass = load_plm("bert", "bert-base-cased")
promptTemplate = ManualTemplate(
    text = prompt_text,
    tokenizer = tokenizer,
)

promptVerbalizer = ManualVerbalizer(
    classes = classes,
    label_words = {
        "2009": ["2009"],
        "2000": ["2000"],
    },
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
        print(classes[preds])
# predictions would be 1, 0 for classes 'positive', 'negative'



 