def build_from_cfg(cfg , registry):
    obj_type = cfg.pop('type')
    obj_cls = registry.module_dict_[obj_type]
    return obj_cls(**cfg)

class Registry:
    """
        registry root to map strings to classes or func
    """
    def __init__(self , name):
        self.name_ = name
        self.module_dict_ = dict()

    def __len__(self):
        return len(self.module_dict_)

    def register_module_(self , module_class):
        module_name = module_class.__name__
        self.module_dict_[module_name] = module_class

    def register_module(self):
        """
        Register a module
        Functions:
            add a record to self.module_dict_ which key is the class name and
        the value is the class
        """
        def _register(module):
            self.register_module_(module_class=module)
            return module

        return _register

    @property
    def name(self):
        return self.name_

    @property
    def mode_dict(self):
        return self.module_dict_