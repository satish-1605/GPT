class ByteEncoder:

    def __init__(self):
        self.byte_encoder = self.bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

    @staticmethod
    def bytes_to_unicode():
        """
        Maps every byte (0-255) to a Unicode character.
        """
        bs = list(range(ord("!"), ord("~") + 1))
        bs += list(range(ord("¡"), ord("¬") + 1))
        bs += list(range(ord("®"), ord("ÿ") + 1))

        cs = bs[:]
        n = 0

        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(256 + n)
                n += 1

        return dict(zip(bs, map(chr, cs)))

    def encode(self, text):
        """
        Convert a UTF-8 string into the GPT-2 byte-level representation.
        """
        return "".join(
            self.byte_encoder[b]
            for b in text.encode("utf-8")
        )

    def decode(self, text):
        """
        Convert the byte-level representation back into a UTF-8 string.
        """
        byte_values = [self.byte_decoder[c] for c in text]
        return bytes(byte_values).decode("utf-8")