import json
from inspect import isgenerator
from typing import Any, Dict, Iterable, Optional, Union


from uphill.core.audio import index_by_id_and_check
from uphill.document import Supervision
from uphill import loggerx

from .mixins import AllMixins


class SupervisionArray(AllMixins):
    
    NAME_TO_SUPERVISIONARRAY = {}
    SUPERVISIONARRAY_TO_NAME = {}
    SAVE_MEMORY: bool = True
    
    # Store anything else the user might want.
    custom: Optional[Dict[str, Any]] = None

    def __init_subclass__(cls, **kwargs):
        if cls.__name__ not in SupervisionArray.NAME_TO_SUPERVISIONARRAY:
            key_name = cls.__name__.lower().replace("supervisionarray", "")
            SupervisionArray.NAME_TO_SUPERVISIONARRAY[key_name] = cls
            SupervisionArray.SUPERVISIONARRAY_TO_NAME[cls] = key_name
        super().__init_subclass__(**kwargs)
    
    def __init__(self, supervisions: Iterable[Supervision] = None) -> None:
        # self.supervisions = index_by_id_and_check(supervisions) if supervisions is not None else {}
        if supervisions is None:
            self.supervisions = {}
        elif isgenerator(supervisions):
            self.supervisions = index_by_id_and_check(supervisions)
        elif len(supervisions) > 0:
            if isinstance(supervisions[0], str):
                self.supervisions = {}
                for supervision_str in supervisions:
                    key = self.decompress_item(supervision_str).id
                    assert key not in self.supervisions, f"Duplicated manifest ID: {key}"
                    self.supervisions[key] = supervision_str
            else:
                self.supervisions = index_by_id_and_check(supervisions)
        else:
            self.supervisions = {}
        
    ################################
    ## property getter and setter ##
    ################################
    @property
    def data(self) -> Union[Dict[str, Supervision], Iterable[Supervision]]:
        """Alias property for ``self.supervisions``"""
        return self.supervisions

    @data.setter
    def data(self, key_value_dict: Dict[str, Supervision]):
        self.supervisions = key_value_dict

    @property
    def values(self) -> Union[Dict[str, Supervision], Iterable[Supervision]]:
        return self.supervisions.values()
    
    @property
    def ids(self) -> Iterable[str]:
        return self.supervisions.keys()
    
    ############################
    ## construction functions ##
    ############################
    @staticmethod
    def from_supervisions(supervisions: Iterable[Supervision]) -> "SupervisionArray":
        return SupervisionArray(supervisions=supervisions)
    
    from_items = from_supervisions
    
    @staticmethod
    def from_dict(data: Iterable[dict]) -> "SupervisionArray":
        assert len(data) > 0
        if SupervisionArray.SAVE_MEMORY:
            return SupervisionArray(
                supervisions=data
            )
        else:
            return SupervisionArray(
                supervisions=(Supervision.from_dict(raw_rec) for raw_rec in data)
            )

    ##################
    ### operations ###
    ##################
    def append(self, supervision: Supervision, force_update: bool = False) -> None:
        if isinstance(supervision, str):
            supervision = self.decompress_item(supervision)
        if supervision.id in self.supervisions:
            if not force_update:
                loggerx.warning(f"Duplicated Supervision ID: {supervision.id}, not update")
                return
            loggerx.warning(f"Duplicated Supervision ID: {supervision.id}, update to new supervision")
        self.supervisions[supervision.id] = self.compress_item(supervision)


    ##########################
    ### internal functions ###
    ##########################
    def _item_from_dict(self, item: Dict):
        return Supervision.from_dict(item)
    
    def __eq__(self, other: 'SupervisionArray') -> bool:
        return self.supervisions == other.supervisions
    
    def __repr__(self):
        return f"{self.__class__.__name__}(len={len(self)})"

    def __contains__(self, item: Union[str, Supervision]) -> bool:
        key = item if isinstance(item, str) else item.id
        return key in self.supervisions

    def __getitem__(self, supervision_id_or_index: Union[int, str]) -> Supervision:
        if isinstance(supervision_id_or_index, int):
            if supervision_id_or_index >= len(self):
                raise IndexError(f"array index out of range, array length is {len(self)}")
            # ~100x faster than list(dict.values())[index] for 100k elements
            supervision = next(
                val 
                for idx, val in enumerate(self) 
                if idx == supervision_id_or_index
            )
        else:
            supervision = self.supervisions[supervision_id_or_index]
        return self.decompress_item(supervision)
    
    def __setitem__(self, supervision_id_or_index: Union[int, str], value):
        if isinstance(supervision_id_or_index, int):
            if supervision_id_or_index >= len(self):
                raise IndexError(f"array index out of range, array length is {len(self)}")
            # ~100x faster than list(dict.values())[index] for 100k elements
            supervision_id = self.decompress_item(
                next(
                    val 
                    for idx, val in enumerate(self) 
                    if idx == supervision_id_or_index
                )
            ).id
        else:
            supervision_id = supervision_id_or_index
        self.supervisions[supervision_id] = value

    def __iter__(self) -> Iterable[Supervision]:
        return iter(self.decompress_item(val) for val in self.values)

    def __len__(self) -> int:
        return len(self.supervisions)
