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