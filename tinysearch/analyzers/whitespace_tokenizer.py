from typing import List


class WhitespaceTokenizer:
    def __call__(self, text: str) -> List[str]:
        return text.split()
