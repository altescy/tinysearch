from typing import Callable, Mapping, Sequence, TypeVar

Analyzer = Callable[[str], Sequence[str]]
Document = TypeVar("Document", bound=Mapping[str, str])
