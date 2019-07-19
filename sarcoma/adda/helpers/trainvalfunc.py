from sys import exit as xit
import re
import inspect
# class _decor(object):
#     """
#     https://python-3-patterns-idioms-test.readthedocs.io/en/latest/PythonDecorators.html#review-decorators-without-arguments
#     No args can be sent to this decorator
#     - __init__ gets myfunc as argument
#     - __call__ gets called when myfunc(asdf) is called. Called is passed any arguments sent to myfunc
#     """
#     def __init__(self, func, *args):
#         self.func=func
#         self.msg="I'm a _decor. Not a Cls!"
#     def __call__(self, *args,**kwargs):
#         print(*args,**kwargs)
#         xit()
#         # print("Called")
#         def wrap(other,arg):
#             print(other.msg)
#             self.func(other, arg)

# class TrMethods(object):
#     """docstring for TrSplitter."""


class InitializationError(Exception):
    pass


class _tvf_splits(object):
    tr = 1
    val = 2
    tst = 3
    iter_list=[tr,val,tst]
    def __iter__(self):
        return iter(self.iter_list)

splits = _tvf_splits

_tvf_split_names = {
    _tvf_splits.tr: "training",
    _tvf_splits.val: "validation",
    _tvf_splits.tst: "test"
}
split_names=_tvf_split_names

_tvf_str2split = {
    "tr": _tvf_splits.tr,
    "val": _tvf_splits.val,
    "tst": _tvf_splits.tst
}


_tvf_split2str = {s: nme for nme, s in _tvf_str2split.items()}


def __TrainValFuncs__getattr__(cls):
    def __getattr__(self, attr):
        # DataSet().dataset_tr -> DataSet().dataset_(str2split['tr'])
        x = re.match(r'(.*?_)(tr|val|tst)$', attr)
        if x:
            func_name = x.group(1)
            split_name = x.group(2)
            split_num = _tvf_str2split[split_name]
            # print(f"Trying to access {func_name}")
            # print(self._tvf_functions)
            if func_name in self._tvf_props and hasattr(self, func_name):
                # If it's a property, we call the function for them
                return getattr(self, func_name)(split=split_num)
            elif func_name in self._tvf_functions and hasattr(self, func_name):
                # If it's a function, we have to provide some arguments;
                return lambda *args, **kwargs: getattr(self, func_name)(split_num,*args, **kwargs)
        return super(cls, self).__getattribute__(attr)
    return __getattr__


def add_to_class(cls, member, name=None):
    if hasattr(member, 'owner_cls'):
        raise ValueError("%r already added to class %r" % (member, member.owner_cls))
    member.owner_cls = cls
    if name is None:
        name = member.__name__
    setattr(cls, name, member)

def _tvf_verify_method(func_name,func):
    if func_name[-1] != "_":
        raise ValueError(f"Function name has to end with an underscore. Got: {func_name}")
    # sign = inspect.signature(func)
    sign=func.signature
    # help(sign)
    splitarg = sign.parameters.get("split")
    sig_names=list(sign.parameters.keys())
    if splitarg is None or "split" not in sig_names or sig_names.index("split")!=1:
        err_txt="Callback function has to have the `split` as the first argument after self."
        if "split" in sig_names:
            err_txt += f"Currently positioned as:{sig_names.index('split')+1}"
        raise ValueError(err_txt)


    if "split" not in sig_names:
        raise ValueError("Split needs to be g")
        sig_names.index("split")
    if sig_names.index("split") != 1:
        print(sig_names.index("split"))



def TrainValFuncs(cls):
    cls._tvf_functions = []
    cls._tvf_props = []
    cls.tvf_splits = _tvf_splits
    setattr(cls, "__getattr__", __TrainValFuncs__getattr__(cls))
    for name, method in cls.__dict__.items():
        if hasattr(method, "_is_tvf_func"):
            _tvf_verify_method(name,method)
            cls._tvf_functions.append(name)
        if hasattr(method, "_is_tvf_prop"):
            cls._tvf_props.append(name)
    return cls


def _trvaldecor(func, prop=False):
    func_name = func.__name__

    def wrapper(self, *args, **kwargs):
        if not hasattr(self, "_tvf_functions"):
            raise InitializationError(f"Whoops. Seems like you didn't use the @TrainValFuncs decorator on the class {self.__class__.__name__}, {self}")
        return func(self, *args, **kwargs)
    wrapper.signature=inspect.signature(func)
    if prop:
        wrapper._is_tvf_prop = True
    else:
        wrapper._is_tvf_func = True

    return wrapper


def trvalprop(func):
    return _trvaldecor(func, prop=True)


def trvalfunc(func):
    return _trvaldecor(func, prop=False)

if __name__ == '__main__':
    @TrainValFuncs
    class Cls(object):
        """docstring for Parent."""
        #pylint: disable=R0201

        def __init__(self):
            self.msg = "I'm a Cls!"

        @trvalprop
        def test_prop_(self, split=splits.tr):
            print(f"Property got split {split}")
            return "PROP"

        @trvalfunc
        def test_func_(self, split, arg):
            print(f"Property got split: {split} and argument: {arg}")
            return "FUNC"
        def not_tvf_(self,arg):
            print(arg)
    c = Cls()
    print(c.test_prop_tr)
    print(c.test_func_val("argument"))
    try:
        print(c.not_tvf_tr("SHOULD FAIL!"))
    except AttributeError as e:
        pass
    print(c.not_tvf_("All good :)"))
