from typing import Any, Optional, Iterable


from uphill import loggerx
from uphill.core.utils import (
    Pathlike,
    Manifest,
    save_yaml, load_yaml,
    save_json, load_json,
    save_jsonl, load_jsonl,
    extension_contains,
    load_manifest
)


class SerializableMixin:
    ###########################
    ### exporting functions ###
    ###########################
    def to_dict(self) -> Iterable[dict]:
        return list(r.to_dict() for r in self)

    def yield_dict(self) -> Iterable[dict]:
        for r in self:
            yield r.to_dict()

    def to_yaml(self, path: Pathlike) -> None:
        save_yaml(self.to_dicts(), path)

    @classmethod
    def from_yaml(cls, path: Pathlike) -> Manifest:
        data = load_yaml(path)
        return cls.from_dict(data)
    
    def to_json(self, path: Pathlike) -> None:
        save_json(self.to_dicts(), path)

    @classmethod
    def from_json(cls, path: Pathlike) -> Manifest:
        data = load_json(path)
        return cls.from_dict(data)
    
    def to_jsonl(self, path: Pathlike) -> None:
        save_jsonl(self.yield_dict(), path)

    @classmethod
    def from_jsonl(cls, path: Pathlike) -> Manifest:
        data = load_jsonl(path)
        return cls.from_dict(data)
    
    def to_file(self, path: Pathlike) -> None:
        if extension_contains(".jsonl", path) or path == "-":
            self.to_jsonl(path)
        elif extension_contains(".json", path):
            self.to_json(path)
        elif extension_contains(".yaml", path):
            self.to_yaml(path)
        else:
            raise ValueError(f"Unknown serialization format for: {path}")

    @classmethod
    def from_file(cls, path: Pathlike, partition: str = "0/1", drop_last: bool = False) -> Manifest:
        loggerx.info(f"loading {path}")
        return load_manifest(path, manifest_cls=cls, partition=partition, drop_last=drop_last)
