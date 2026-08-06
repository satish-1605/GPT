from datasets import load_dataset

from dotenv import load_dotenv
import os
load_dotenv(override=True)
hf_token = os.getenv("HF_TOKEN")


ds = load_dataset("roneneldan/TinyStories", token=hf_token)

# print(ds)
# print(len(ds["train"]))
# print(ds["train"].column_names)
# # print(ds["train"][0])
# # print(ds["train"][0]["text"])

# for i in range(3):
#     print("=" * 100)
#     print(ds["train"][i]["text"])
