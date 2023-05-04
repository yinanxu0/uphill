from typing import Any, Union
from rich.tree import Tree

from uphill.core.utils import get_args


class PlotMixin:
    def summary(self) -> str:
        """Print the structure and attribute summary of this DocumentArray object.

        .. warning::
            Calling {meth}`.summary` on large DocumentArray can be slow.

        """

        """Print non-empty fields and nested structure of this Document object."""        
        info_tree = self._summary_helper(None, self.__class__.__name__, self)
        from rich import print
        print(info_tree)
    
    def _summary_helper(self, tree: Tree, name: str, obj: Any) -> Tree:
        from uphill import Document, DataSource, Supervision
        MANIFEST = Union[Document, DataSource, Supervision]
        MANIFEST_TYPE = get_args(MANIFEST)
        
        if tree is None:
            tree = Tree(name)
        
        if isinstance(obj, MANIFEST_TYPE):
            # MANIFEST
            obj_tree = Tree(label=f":{obj.__class__.__name__}:",)
            obj_tree.add(f':id: [b]{obj.id}[/b]')
            for key in obj.__dataclass_fields__.keys():
                if key in ["id"] or obj.__getattribute__(key) is None:
                    continue
                sub_obj = obj.__getattribute__(key)
                if isinstance(sub_obj, MANIFEST_TYPE):
                    # MANIFEST
                    sub_obj_tree = self._summary_helper(None, key, sub_obj)
                    obj_tree.add(sub_obj_tree)
                elif isinstance(sub_obj, list) and len(sub_obj) > 0 and isinstance(sub_obj[0], MANIFEST_TYPE):
                    # Iterable[MANIFEST]
                    for idx, sub_sub_obj in enumerate(sub_obj):
                        sub_sub_obj_tree = self._summary_helper(None, f"{key}-[{idx}]", sub_sub_obj)
                        obj_tree.add(sub_sub_obj_tree)
                else:
                    obj_tree.add(f':{key}: [b]{str(sub_obj)}[/b]')
            tree.add(obj_tree)
        elif isinstance(obj, list) and len(obj) > 0 and isinstance(obj[0], MANIFEST_TYPE):
            # Iterable[MANIFEST]
            for sub_obj in obj:
                sub_tree = self._summary_helper(None, sub_obj.__class__.__name__, sub_obj)
                tree.add(sub_tree)
        else:
            tree.add(f':{name}: [b]{str(obj)}[/b]')
        return tree
