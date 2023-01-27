import fcntl
import json
import pickle
from contextlib import contextmanager
from io import BytesIO
from os import PathLike
from pathlib import Path
from typing import BinaryIO, Dict, Generic, Iterator, NamedTuple, Optional, Tuple, TypedDict, TypeVar, Union, cast

from tinysearch.storages.storage import Storage

T = TypeVar("T")


class Metadata(TypedDict):
    pagesize: int


class Key(NamedTuple):
    value: str

    def __str__(self) -> str:
        return self.value

    def to_bytes(self) -> bytes:
        value = self.value.encode("utf-8")
        length = len(value).to_bytes(4, "little")
        return length + value

    @classmethod
    def from_binaryio(cls, f: BinaryIO) -> "Key":
        length = int.from_bytes(f.read(4), "little")
        return cls(f.read(length).decode("utf-8"))


class Index(NamedTuple):
    page: int
    offset: int
    length: int

    def to_bytes(self) -> bytes:
        return self.page.to_bytes(4, "little") + self.offset.to_bytes(4, "little") + self.length.to_bytes(4, "little")

    @classmethod
    def from_binaryio(cls, f: BinaryIO) -> "Index":
        return cls(
            int.from_bytes(f.read(4), "little"),
            int.from_bytes(f.read(4), "little"),
            int.from_bytes(f.read(4), "little"),
        )


class FileStorage(Storage[T]):
    def __init__(
        self,
        path: Union[str, PathLike],
        pagesize: Optional[int] = None,
    ) -> None:
        self._path = Path(path)
        self._pagesize = pagesize
        self._indices: Dict[Key, Index] = {}
        self._pageios: Dict[int, BinaryIO] = {}

        self._path.mkdir(parents=True, exist_ok=True)
        index_filename = self._get_index_filename()
        if not index_filename.exists():
            index_filename.touch()

        metadata_filename = self._get_metadata_filename()
        if metadata_filename.exists():
            self._load_metadata()
        else:
            self._save_metadata()

        self._indexio: BinaryIO = index_filename.open("rb+")
        if self._indexio.seek(0, 2) > 0:
            self._load_indices()

        for page, page_filename in self._iter_page_filenames():
            self._pageios[page] = page_filename.open("rb+")

    @contextmanager
    def lock(self) -> Iterator[None]:
        lockfile = self._get_lock_filename().open("w")
        try:
            fcntl.flock(lockfile, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(lockfile, fcntl.LOCK_UN)
            lockfile.close()

    def _get_lock_filename(self) -> Path:
        return self._path / "storage.lock"

    def _get_metadata_filename(self) -> Path:
        return self._path / "metadata.json"

    def _get_index_filename(self) -> Path:
        return self._path / "index.bin"

    def _get_page_filename(self, page: int) -> Path:
        return self._path / f"page_{page:08d}"

    def _iter_page_filenames(self) -> Iterator[Tuple[int, Path]]:
        for page_filename in self._path.glob("page_*"):
            page = int(page_filename.stem.split("_", 1)[1])
            yield page, page_filename

    def _load_metadata(self) -> None:
        metadata_filename = self._get_metadata_filename()
        if not metadata_filename.exists():
            raise FileNotFoundError(metadata_filename)
        with metadata_filename.open("r") as f:
            metadata = json.load(f)
        self._pagesize = metadata["pagesize"]

    def _save_metadata(self) -> None:
        metadata_filename = self._get_metadata_filename()
        with metadata_filename.open("w") as f:
            json.dump({"pagesize": self._pagesize}, f)

    def _load_indices(self) -> None:
        eof = self._indexio.seek(0, 2)
        self._indexio.seek(0)
        while self._indexio.seek(0, 1) < eof:
            key = Key.from_binaryio(self._indexio)
            index = Index.from_binaryio(self._indexio)
            self._indices[key] = index

    def _add_index(self, key: Key, index: Index) -> None:
        self._indices[key] = index
        key_value = key.to_bytes()
        index_value = index.to_bytes()
        self._indexio.seek(0, 2)
        self._indexio.write(key_value)
        self._indexio.write(index_value)

    def _encode(self, value: T) -> bytes:
        buffer = BytesIO()
        pickle.dump(value, buffer)
        return buffer.getvalue()

    def _decode(self, value_bytes: bytes) -> T:
        return cast(T, pickle.loads(value_bytes))

    def __contains__(self, key: Union[str, Key]) -> bool:
        if isinstance(key, str):
            key = Key(key)
        return key in self._indices

    def __getitem__(self, key: Union[str, Key]) -> T:
        if isinstance(key, str):
            key = Key(key)
        if key not in self._indices:
            raise KeyError(key)
        index = self._indices[key]
        pageio = self._pageios[index.page]
        pageio.seek(index.offset)
        return cast(T, pickle.loads(pageio.read(index.length)))

    def __setitem__(self, key: Union[str, Key], value: T) -> None:
        if isinstance(key, str):
            key = Key(key)

        if key in self._indices:
            raise KeyError(f"Key {key.value} already exists")

        encoded_value = self._encode(value)
        length = len(encoded_value)

        pageio: BinaryIO
        if not self._pageios:
            page = 0
            offset = 0
            pageio = self._get_page_filename(page).open("wb+")
            self._pageios[page] = pageio
        else:
            page = len(self._pageios) - 1
            pageio = self._pageios[page]

        offset = pageio.seek(0, 2)
        if self._pagesize is not None and offset + length > self._pagesize:
            page += 1
            offset = 0
            pageio = open(self._get_page_filename(page), "wb+")
            self._pageios[page] = pageio

        pageio.write(encoded_value)
        self._add_index(key, Index(page, offset, length))

    def __len__(self) -> int:
        return len(self._indices)

    def __iter__(self) -> Iterator[T]:
        yield from self.values()

    def keys(self) -> Iterator[str]:
        for key in self._indices:
            yield key.value

    def values(self) -> Iterator[T]:
        for key in self._indices:
            yield self[key]

    def items(self) -> Iterator[Tuple[str, T]]:
        for key in self._indices:
            yield key.value, self[key]

    def flush(self) -> None:
        self._indexio.flush()
        for pageio in self._pageios.values():
            pageio.flush()

    def close(self) -> None:
        self._indexio.close()
        for pageio in self._pageios.values():
            pageio.close()
