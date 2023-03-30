
import sys
import transformers
import coremltools

coremltools.libmilstoragepython


# model_name = "distilbert-base-uncased-finetuned-sst-2-english"


# tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
# tokenized = tokenizer(
#     ["Sample input text to trace the model"],
#     return_tensors="pt",
#     max_length=128,  # token sequence length
#     padding="max_length",
# )
# for name, tensor in tokenized.items():
#     print(name)
#     print(tensor)
#     # sys.exit()
# print(name)