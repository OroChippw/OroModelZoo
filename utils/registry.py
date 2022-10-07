import inspect


def build_from_cfg(cfg , registry ,args = None):
    '''
    Func:
        build class from cfg
    Args:
        cfg:Config dict which should at least contain the key "type"
        registry:registry object

    Returns:
        an instantiated object
    '''
    if not isinstance(cfg , dict):
        raise TypeError(f'cfg must be a dict , but got {type(cfg)}.')
    if 'type' not in cfg:
        raise KeyError(f'`cfg` must contain the key `type`')
    if not (isinstance(args , dict) or args is None):
        raise TypeError(f'args must be a dict or None , but got {type(args)}')

    cfg_ = cfg.copy()

    if args is not None:
        for key , value in args.item():
            cfg_.setdefault(key , value)

    obj_type = cfg_.pop('type')
    if isinstance(obj_type , str):
        obj_cls = registry.get(obj_type)
        if obj_cls is None :
            raise KeyError(
                f'{obj_type} is not in the {registry.name} registry')
    elif inspect.isclass(obj_type):
        obj_cls = obj_type
    else :
        raise TypeError(
            f'`obj_type` must be a str or valid type , but got {type(obj_type)}'
        )
    return obj_cls(**cfg)

class Registry:
    """
        registry root to map strings to classes or func
    """
    def __init__(self , name):
        self.name_ = name # class string name
        self.module_dict_ = dict() # a dict , Registered modules will be stored in this dictionary type variable

    def __len__(self):
        return len(self.module_dict_)

    def __contains__(self, item):
        return self.get(item) is not None

    def __repr__(self):
        format_str = self.__class__.__name__ + \
            f'(name = {self.name_} , module_dict = {self.module_dict_}'
        return format_str

    def register_module_(self , module_class , module_name = None):
        if not inspect.isclass(module_class):
            raise TypeError(f'module must be a class , but got {type(module_class)}.')
        if module_name is None:
            module_name = module_class.__name__
        if module_name in self.module_dict_:
            raise KeyError(f'{module_name} is already registered in {self.name_}')
        # 在module中dict新增key和value。key为类名，value为类对象
        self.module_dict_[module_name] = module_class

    def register_module(self):
        """
        Register a module
        Functions:
            add a record to self.module_dict_ which key is the class name and
        the value is the class
        Memo:
            register_module()先运行，然后返回内层的_register() 函数作为真正的装饰器，去接收类的对象进行注册
        """
        def _register(module):
            self.register_module_(module_class=module)
            return module

        return _register

    @property
    def name(self):
        # use @property to implement read-only func
        return self.name_

    @property
    def mode_dict(self):
        return self.module_dict_

    def get(self , key):
        '''
        Func:
            Get the registry record
        Args:
            key:the class name in string format
        Returns:
            the corresponding class
        '''
        return self.module_dict_.get(key , None)
