from typing import Any, Tuple
from dataclasses import fields

from uphill.document.helper import ifnone


class InternalMixin:
    # Store anything else the user might want.
    # custom: Optional[Dict[str, Any]] = None
    
    def __setattr__(self, key: str, value: Any):
        """
        This magic function is called when the user tries to set an attribute.
        We use it as syntactic sugar to store custom attributes in ``self.custom``
        field, so that they can be (de)serialized later.
        """
        if key in self.__dataclass_fields__:
            super().__setattr__(key, value)
        else:
            custom = ifnone(self.custom, {})
            custom[key] = value
            self.custom = custom

    def __getattr__(self, name: str) -> Any:
        """
        This magic function is called when the user tries to access an attribute
        of object that doesn't exist. It is used for accessing the custom
        attributes of cuts.

        We use it to look up the ``custom`` field: when it's None or empty,
        we'll just raise AttributeError as usual.

        Example of attaching and reading an element::

            >>> doc = Document.from_uri('123-5678.wav')
            >>> doc.alpha = 0.99
            >>> doc.alpha
            0.99

        """
        if self.custom is None:
            raise AttributeError(f"No such attribute: {name}")
        if name in self.custom:
            # Somebody accesses raw Document manifest or wrote 
            # a custom piece of metadata into Document.
            return self.custom[name]
        raise AttributeError(f"No such attribute: {name}")
    
    def _non_empty_fields(self) -> Tuple[str]:
        r = []
        for f in fields(self):
            f_name = f.name
            v = getattr(self, f_name)
            if v is not None:
                r.append(f_name)
        return tuple(r)
    
    def __repr__(self):
        content = str(self._non_empty_fields())
        content += f' at {getattr(self, "id", id(self))}'
        return f'<{self.__class__.__name__} {content.strip()}>'

