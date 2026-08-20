class CorpusTokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def tokenize_documents(self, documents:list[str])->list[list[int]]:
        eos_token = "<|endoftext|>"
        return [self.tokenizer.encode(document + eos_token) for document in documents] 


    def build_token_stream(
            self, tokenized_documents:list[list[int]]
        )-> list[int]:

        token_stream = []

        for document in tokenized_documents:
            token_stream.extend(document)

        return token_stream
        

    
# preprocessor = DatasetPreprocessor()
# config = GPTConfig()
# documents = preprocessor.preprocess_corpus(
#         input_file=config.fineweb
#     )
# # print(documents[0])
# train_docs, val_docs, test_docs = train_val_test_split(
#         documents
#     )
# tokenizer = BPETokenizer.from_pretrained(
#     "artifacts/tokenizer"
# )

# corpus_tokenizer = CorpusTokenizer(tokenizer)

# train_ids = corpus_tokenizer.tokenize_documents(train_docs)
# val_ids = corpus_tokenizer.tokenize_documents(val_docs)
# test_ids = corpus_tokenizer.tokenize_documents(test_docs)

# # print(train_ids[0:2])

# print("train documents", len(train_docs))
# print("Train token sequences", len(train_ids))
# print("val documents", len(val_docs))
# print("val token sequences", len(val_ids))
# print("test documents", len(test_docs))
# print("test token sequences", len(test_ids))
# token_stream = corpus_tokenizer.build_token_stream(train_ids)
# print("length of token streams", len(token_stream))