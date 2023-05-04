import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Union, Callable


from uphill.core.utils import asdict_nonull

from .document import Document
from .mixins import AllMixin


@dataclass(repr=False, eq=False)
class Supervision(AllMixin):
    source: Document
    target: Document
    id: str = field(default_factory=lambda: os.urandom(12).hex())
    # Store anything else the user might want.
    custom: Optional[Dict[str, Any]] = None
    
    ################################
    ## property getter and setter ##
    ################################
    
    ############################
    ## construction functions ##
    ############################
    @staticmethod
    def from_document(
        source: Document, 
        target: Document, 
        id: Optional[Union[str, Callable[[], str]]] = None,
    ) -> "Supervision":
        id = (
            os.urandom(12).hex()
            if id is None
            else id()
            if callable(id)
            else id
        )
        return Supervision(source=source, target=target, id=id)
        
    @staticmethod
    def from_dict(data: dict) -> "Supervision":
        assert "source" in data
        assert "target" in data
        source = Document.from_dict(data.pop("source"))
        target = Document.from_dict(data.pop("target"))
        return Supervision(source=source, target=target, **data)


    ###########################
    ### exporting functions ###
    ###########################
    def to_dict(self) -> dict:
        doc_info = asdict_nonull(self)
        doc_info['source'] = self.source.to_dict()
        doc_info["target"] = self.target.to_dict()
        return doc_info


    ##################
    ### operations ###
    ##################

    ##########################
    ### internal functions ###
    ##########################
    def __eq__(self, other: 'Supervision') -> bool:
        if self.id != other.id:
            return False 
        if self.source != other.source:
            return False
        if self.target != other.target:
            return False
        return True

