import string
from typing import List


class SimpleAnalyzer:
    def __call__(self, text: str) -> List[str]:
        text = text.lower()
        for c in string.punctuation:
            text = text.replace(c, " ")
        return text.split()
