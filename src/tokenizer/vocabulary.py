class Vocabulary:
    def __init__(self, tokens):
        self.token_to_id = {}
        for token in tokens:
            if token not in self.token_to_id:
                self.token_to_id[token] = len(self.token_to_id)

        self.id_to_token = {v:k for k,v in self.token_to_id.items()}
                
    def add_token(self, token):
        if token in self.token_to_id:
            return self.token_to_id[token]

        new_id = len(self.token_to_id)
        self.token_to_id[token] = new_id

        self.id_to_token[new_id] = token
        return  new_id

    def get_id(self, token):
        if token in self.token_to_id:
            return self.token_to_id.get(token)
        else:
            raise ValueError(f"token {token} not found")


    def get_token(self, id):
            if id in self.id_to_token:
                return self.id_to_token.get(id)
            else:
                raise ValueError(f"id {id} not found")

    def contains(self, token):
        return token in self.token_to_id

    def __len__(self):
        return len(self.token_to_id)