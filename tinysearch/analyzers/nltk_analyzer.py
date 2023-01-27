import string
from typing import Callable, List, Optional, Sequence, Tuple, cast


class NltkAnalyzer:
    def __init__(
        self,
        stopwords: Sequence[str] = (),
        ngram_range: Tuple[int, int] = (1, 1),
        lowercase: bool = True,
        remove_punctuation: bool = True,
        stemmer: Optional[str] = "porter",
    ) -> None:
        from nltk import word_tokenize

        self._stopwords = set(stopwords)
        self._ngram_range = ngram_range
        self._lowercase = lowercase
        self._remove_punctuation = remove_punctuation
        self._tokenizer = cast(Callable[[str], List[str]], word_tokenize)

        if stemmer == "porter":
            from nltk.stem.porter import PorterStemmer

            self._stemmer = PorterStemmer()
        elif stemmer == "lancaster":
            from nltk.stem.lancaster import LancasterStemmer

            self._stemmer = LancasterStemmer()
        elif stemmer == "snowball":
            from nltk.stem.snowball import SnowballStemmer

            self._stemmer = SnowballStemmer("english")
        elif stemmer == "wordnet":
            from nltk.stem import WordNetLemmatizer

            self._stemmer = WordNetLemmatizer()
        elif stemmer is None:
            self._stemmer = None
        else:
            raise ValueError(f"Unknown stemmer: {stemmer}")

    def _ngramize(self, tokens: Sequence[str]) -> List[str]:
        ngrams = []
        for n in range(self._ngram_range[0], self._ngram_range[1] + 1):
            ngrams.extend(["##".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)])
        return ngrams

    def __call__(self, text: str) -> List[str]:

        if self._lowercase:
            text = text.lower()
        if self._remove_punctuation:
            text = text.translate(str.maketrans("", "", string.punctuation))

        # tokenize text
        tokens = self._tokenizer(text)

        # stem tokens if stemmer is provided
        if self._stemmer is not None:
            tokens = [self._stemmer.stem(token) for token in tokens]

        # remove stopwords
        tokens = [token for token in tokens if token not in self._stopwords]

        # ngramize tokens
        if self._ngram_range is not None:
            tokens = self._ngramize(tokens)

        return tokens
