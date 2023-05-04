from typing import Any, Optional


from uphill.core.utils import (
    Pathlike,
    Manifest,
    save_yaml, load_yaml,
    save_json, load_json,
    extension_contains,
    load_manifest
)
from uphill import loggerx


class SerializableMixin:
    def to_yaml(self, path: Pathlike) -> None:
        save_yaml(self.to_dict(), path)

    @classmethod
    def from_yaml(cls, path: Pathlike) -> Manifest:
        data = load_yaml(path)
        return cls.from_dict(data)
    
    def to_json(self, path: Pathlike) -> None:
        save_json(list(self.to_dict()), path)

    @classmethod
    def from_json(cls, path: Pathlike) -> Manifest:
        data = load_json(path)
        return cls.from_dict(data)
    
    def to_file(self, path: Pathlike) -> None:
        if extension_contains(".json", path):
            self.to_json(path)
        elif extension_contains(".yaml", path):
            self.to_yaml(path)
        else:
            raise ValueError(f"Unknown serialization format for: {path}")

    @classmethod
    def from_file(cls, path: Pathlike) -> Manifest:
        return load_manifest(path, manifest_cls=cls)
    
    # TODO: 
    '''
    def to_bytes(self) -> bytes:
        ...
    
    @classmethod
    def from_bytes(cls, data: bytes) -> Manifest:
        ...
    
    '''
