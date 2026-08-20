
class BPE:
    def __init__(self, merges):
        """
        Store BPE merge pairs with their priority/rank.
        Lower rank = higher priority.
        """

        self.bpe_ranks = {}
        for rank, pair in enumerate(merges):
            self.bpe_ranks[pair] = rank

    def get_pairs(self, word) -> None:
        """
        Return all unique adjacent symbol pairs.
        """
        pairs = set()
        if len(word) < 2:
            return pairs
        
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pairs.add(pair)

        return pairs

    def apply_bpe(self, word):
        """
        Apply BPE merge rules repeatedly until
        no applicable merge remains.
        """

        word = tuple(word)

        while True:
            best_pair = None
            best_rank = float("inf")

            pairs = self.get_pairs(word)

            if not pairs:
                break

            for pair in pairs:
                if pair in self.bpe_ranks:
                    rank = self.bpe_ranks[pair]

                    if rank < best_rank:
                        best_rank = rank
                        best_pair = pair

            if best_pair is None:
                break
            
            i = 0
            new_word = []
            while i < len(word):
                if (i < len(word) -1 
                    and (word[i], word[i+1]) == best_pair):

                    merged_symbols = word[i] + word[i+1]
                    new_word.append(merged_symbols)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            word = tuple(new_word)
        return word