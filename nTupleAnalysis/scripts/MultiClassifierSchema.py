# Based on https://github.com/CoffeaTeam/coffea/blob/7dd4f863837a6319579f078c9e445c61d9106943/coffea/nanoevents/schemas/nanoaod.py
from coffea.nanoevents.schemas.base import BaseSchema, zip_forms

class MultiClassifierSchema(BaseSchema):
    """Basic multiclassifier friend tree schema"""
    def __init__(self, base_form, name=''):
        super().__init__(base_form)
        self.mixins = {}
        self._form["contents"] = self._build_collections(self._form["contents"])

    def _build_collections(self, branch_forms):
        for k in branch_forms:
            if k.startswith('SvB_MA'):
                name = 'SvB_MA'
                break
            if k.startswith('SvB'):
                name = 'SvB'
                break
            if k.startswith('FvT'):
                name = 'FvT'
                break

        mixin = self.mixins.get(name, "NanoCollection")

        # simple collection
        output = {}
        output[name] = zip_forms(
            {
                k[len(name) + 1 :]: branch_forms[k]
                for k in branch_forms
                if k.startswith(name + "_")
            },
            name,
            record_name=mixin,
        )
        output[name].setdefault("parameters", {})
        output[name]["parameters"].update({"collection_name": name})

        return output            

    @property
    def behavior(self):
        """Behaviors necessary to implement this schema"""
        from coffea.nanoevents.methods import nanoaod

        return nanoaod.behavior
