import json
from typing import Union, Iterable, List, Optional


from uphill.core.utils import T
from uphill import loggerx


class AutoCompressMixin:
    """
    Helper base class with methods that compress items to save memory
    on manifest classes such as `uphill.DocumentArray`, 
    `uphill.SupervisionArray`, etc.
    """
    def decompress(self, ids: Union[int, str, list]):
        if isinstance(ids, int) or isinstance(ids, str):
            ids = [ids]
        for id in ids:
            item = self[id]
            if item is None:
                loggerx.warning(f"{id} not in {type(self)}")
                continue
            self[id] = self.decompress_item(item)
    
    def compress(self, ids):
        if isinstance(ids, int) or isinstance(ids, str):
            ids = [ids]
        for id in ids:
            item = self[id]
            if item is None:
                loggerx.warning(f"key {id} not in {type(self)}")
                continue
            self[id] = self.compress_item(item)

    def compress_item(self, item) -> str:
        if isinstance(item, str):
            return item
        return json.dumps(item.to_dict())

    def decompress_item(self, item) -> T:
        if not isinstance(item, str):
            return item
        return self._item_from_dict(json.loads(item))
        
