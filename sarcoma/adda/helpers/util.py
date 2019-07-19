import os
import time
import pickle
import sys
from sys import exit as xit
import click
from termios import tcflush, TCIOFLUSH, TCIFLUSH
import prettytable
import cprints
import termios
import atexit
def relpath(path):
    return os.path.normpath(os.path.join(
        os.path.dirname(os.path.realpath(__file__)), path))


def script_relpath(path):
    return os.path.normpath(os.path.join(
        os.path.dirname(os.path.realpath(caller_id(1)[1])), path))


TICKTOCK_START = 0

# print(os.path.realpath(__file__))
import traceback as tb
import re
def flush_stdin():
    tcflush(sys.stdin, TCIFLUSH)

def merge_dicts(*dicts):
    fd = {}
    for d in dicts:
        fd.update(d)
    return fd

def ensure_list(l):
    if isinstance(l, list):
        return l
    elif isinstance(l, tuple):
        return list(l)
    return [l]
def l2s(lst):
    return [str(a) for a in lst]

def caller_id(offset=0, num_steps=1, real_files_only=False):
    stack = tb.extract_stack()
    caller_info = stack[::-1][1 + offset:]
    if num_steps == 1:#pylint disable: R1705
        c_i=caller_info[0]
        file_path, line, dest = c_i[0], c_i[1], c_i[2]
        file_name = os.path.basename(file_path)
        return file_name, file_path, line
    else:
        to_return=[]
        for c_i in caller_info:
            if num_steps and num_steps>0 and len(to_return) >= num_steps:
                break
            file_path, line, dest=c_i[0], c_i[1], c_i[2]
            file_name = os.path.basename(file_path)
            if real_files_only:
                if not os.path.isfile(file_path):
                    continue

            to_return.append(",".join(l2s([file_name, file_path, line])))
        # for _i in range(num_steps):
            # i=_i*3
            # try:
            #     file_path, line, dest = caller_info[i], caller_info[i+1], caller_info[i+2]
            #     file_name = os.path.basename(file_path)
            #     to_return.append(",".join([file_name, file_path, line]))
            # except IndexError:
            #     break
        return "\n".join(to_return)



def print_stack(verbose=1):
    stack = tb.extract_stack()
    # Make last call on top
    stack = stack[::-1]
    # Remove call to utils (this file)
    stack = stack[1:]
    # Clean stack
    clean_stack = []
    so = stack[0]
    tbl_fields = {
        3: ["filename", "line", "lineno", "locals", "name"],
        2: ["filename", "lineno", "name"],
        1: ["filename", "lineno", "name"]
    }
    tbl_fields[4] = tbl_fields[3]
    tbl_fields[1] = tbl_fields[2]
    field_names = tbl_fields[verbose]
    tbl = prettytable.PrettyTable(field_names=field_names)
    tbl.align = "l"
    for s in stack:
        if not s.line and verbose <= 3:
            continue
        if 2 < verbose <= 4:
            stack_info = [
                s.filename,
                s.line,
                s.lineno,
                s.locals,
                s.name
            ]

            # joiner="\n"
        elif verbose == 2:
            stack_info = [s.filename, s.lineno, s.name]
        elif verbose == 1:
            shortname = os.path.relpath(s.filename)
            if shortname.count("../") > 4:
                shortname = ".../" + os.path.join(os.path.basename(os.path.dirname(s.filename)), os.path.basename(s.filename))
            stack_info = [shortname, s.lineno, s.name]
        tbl.add_row(stack_info)
    print(tbl)


def dcm_file_to_path(file_path):
    # Retuns
    _, file_extension = os.path.splitext(file_path)
    if file_extension == ".dcm":
        return os.path.dirname(file_path)
    return file_path


def path_to_dcm_file(path):
    # Retuns
    if os.path.isdir(path):
        return os.path.join(path, "000000.dcm")
    return path


class BracketChoice(click.Choice):
    def __init__(self, choices, case_sensitive=True):
        super().__init__(choices, case_sensitive=case_sensitive)
        self.letter_choices = []
        for cs in self.choices:
            m = re.search(r"\[([^\]]+)\]", cs)
            if m is None:
                choice = cs
            else:
                choice = m.group(1)
            self.letter_choices.append(choice)

        # self.choices = [ re.replace("(.*?)","",c) for c in choices ]
        self.case_sensitive = case_sensitive

    def convert(self, value, param, ctx):
        # Exact match
        if value in self.letter_choices:
            return value
        return super().convert(value, param, ctx)


def ask_choices(text, choices, choice_type=BracketChoice,
                force_choice=True, **kwargs):
    # https://click.palletsprojects.com/en/7.x/api/#click.prompt
    """
     The choice_type argument is to determine whether to parse brackets ([]) in the choices.
     Usage:
     > result=ask_choices("Do you want icecream or soda?", ["[i]cecream","[s]oda"], default="i")
     >> Do you want icecream or soda? ([i]cream, [s]oda) [i]
     # User now can enter i or s (or hit enter to choose icecream for default)
     # and result now contains the letter chosen
    """
    while True:
        tcflush(sys.stdin, TCIFLUSH)
        prompt_kwargs = dict(confirmation_prompt=False, show_choices=True)
        prompt_kwargs.update(kwargs)
        if not hasattr(choices, '__iter__') or isinstance(choices, str):
            choices = list(choices)
        try:
            response = click.prompt(
                text, type=choice_type(choices), **prompt_kwargs)
            return response
        except click.Abort as e:
            if not force_choice:
                raise
            else:
                sys.stdout.write("\n")
                sys.stdout.flush()

def anykey_timeout(timeout, return_char=False):
    """
        Returns char if char entered, else returns False
        True
        (Returns false if user did not intervene)
    """
    tcflush(sys.stdin, TCIFLUSH)
    start = time.time()
    nb_char = ""
    while True:
        nb_char = read_stdin()
        # if start+sleep <= time.time() or nb_char:
        if start + timeout <= time.time() or nb_char:
            # if nb_char and not return_char:
                # return True
            return True if nb_char else False
            # return False
            # elif nb_char and return_char:
            #     return nb_char
            # elif not nb_char:
            #     return False
        time.sleep(0.1)
    return False
    # if nb_char:
        # do_dialog = True
        # color_print(f"Not saving. Getting dialog", style="notice")

def ask_user(text, default=False, y_args=[],
             n_args=[], force_choice=False, force_ask=False, abort=False, **kwargs):
    """ Wrapper for click.confirm """
        # anykey_timeout
    noask_arg="--no-ask"
    tcflush(sys.stdin, TCIOFLUSH)
    # kwargs could be (from the click docs)
    # abort=False, prompt_suffix=': ', show_default=True, err=False
    # Backwards compability
    if isinstance(default, str):
        if default.lower() == "n":
            default = False
        elif default.lower() == "y":
            default = True
    do_ask=True
    answer=default
    if set(y_args).intersection(set(sys.argv)):
        answer = True
        do_ask=False
    elif set(n_args).intersection(set(sys.argv)):
        answer = False
        do_ask=False

    if noask_arg in sys.argv:
        do_ask=False
        cprints.color_print("AUTO MODE: --complete-auto given.", style="warning")
        cprints.color_print("Would have asked", style="warning")
        print(text)
        cprints.color_print(f"Answering {'y' if answer else 'n'} in 3 seconds", style="notice")
        do_ask=anykey_timeout(3)

    def offer_abort():
        halt_secs=4
        cprints.color_print("WANTED TO ABORT. NOT ABORTING.", syle="danger")
        print(f"Halting {halt_secs} seconds. Any key to abort")
        kill=anykey_timeout(halt_secs)
        if kill:
            raise click.Abort("User aborted.")
    if do_ask:
        while True:
            try:
                answer=click.confirm(text=text, default=default, abort=abort, **kwargs)
                break
            except click.Abort as e:
                if noask_arg in sys.argv:
                    offer_abort()
                if not force_choice:
                    raise
                else:
                    sys.stdout.write("\n")
                    sys.stdout.write("You have to provide an answer!\n")
                    sys.stdout.flush()
    if noask_arg in sys.argv and abort and not answer:
        offer_abort()
    elif abort and not answer:
        raise click.Abort("Got negative answer -- aborting")

    return answer


def _ask_user(msg, default="y", y_args=[], n_args=[],
              response=None, listen_sysargs=True):
    default = default.lower()
    test_for = "n" if default == "y" else "y"
    for yarg in y_args:
        if yarg in sys.argv:
            return True
    for narg in n_args:
        if narg in sys.argv:
            return False
    if "-n" in sys.argv and listen_sysargs:
        response = "n"
    elif "-y" in sys.argv and listen_sysargs:
        response = "y"

    if response is None:
        default_prompt = "([y]|n)" if default == "y" else "(y|[n])"
        response = input("{} {}:".format(msg, default_prompt)).lower()
    if response == test_for:
        # We have default "y" and got "n"
        # Or we have default "n" and got "y"
        return response == "y"
    else:
        # We didn't get something contradictory
        return default == "y"
class ListChooseOne(click.Choice):
    def __init__(self, lst, names=None, message=""):
        self.message=message
        self.lst=lst
        self.names=names
        self.first_letters=[]
        self.min=1
        self.max=len(lst)
    # def set_letters(self):

    def name_for(self, _nopos=None, idx=None, itm=None):
        if _nopos is not None:
            raise ValueError("name_for should not get positional argument!")
        if (idx is None and itm is None) or (itm is not None and idx is not None):
            raise ValueError("ONE of `idx` and `itm` is required in name_for!")

        if idx is not None:
            itm = self.lst[idx]
        elif itm is not None:
            idx = self.lst.index(itm)
        if callable(self.names):
            name = self.names(itm)
        elif isinstance(self.names, (tuple, list)) and idx < len(self.names):
            name = self.names[idx]
        else:
            name = str(itm)
        return name

    def get_message(self):
        prompt_message = self.message
        for i, l in enumerate(self.lst, start=1):
            name=self.name_for(itm=l)
            prompt_message += f"\n{i}: {name}"
        prompt_message += "\nYou can only choose one"
        return prompt_message

    def convert(self, value, param, ctx):
        # Exact match
        chosen_idx=None
        if value.isdigit():
            digit = int(value)
            chosen_idx=digit - 1  # One indexed choices
        elif value.lower() in self.letter_choices:
            chosen_idx=self.letter_choices.index(value)
        else:
            self.fail(f"Invalid choice `{value}` found")
        return chosen_idx,self.lst[chosen_idx]

def ask_choose_one(lst, message=None, names=None, failsafe=True, sysarg_key=None, flush=True):
    """
    lst: Iterable (tuple|int) to choose from
    names: Names (None|Callable|list) to hint what you are choosing (else str of objs in list)
    allow_all: Whether or not the user can choose "All"
    allow_none: Whether or not the user can chose "None"
    combos: (combo_name, [index_list])] # NOTE THAT THIS IS INDEX NOT CHOICE!!
    returns index_of_chosen: int, lst_object(s)_chosen: list
    """
    if sysarg_key is not None and sysarg_key in sys.argv:
        sysarg_value_idx=sys.argv.index(sysarg_key)+1
        sysarg_name=sys.argv[sysarg_value_idx]
        tmp_names=names
        if names is None:
            tmp_names=[str(itm) for itm in lst]
        # if sysarg_name in lst:
        # lst_name_for(lst, names, itm=)
        if sysarg_name in tmp_names:
            idx=tmp_names.index(sysarg_name)
            obj=lst[idx]
            return idx,obj
        else:
            raise click.BadParameter(f"Cannot find config with name {sysarg_name} ({sysarg_key})")
    try:
        if flush:
            tcflush(sys.stdin, TCIFLUSH)
        choices = ListChooseOne(lst, message=message, names=names)
        idx, obj = click.prompt(choices.get_message(), type=choices, confirmation_prompt=False, show_choices=False)
    except SystemExit:
        os._exit(0)
    except click.Abort:
        raise
    except Exception as e:
        if not failsafe:
            raise
        else:
            import time
            print("ERROR DOING LIST ASK! TRIGGERING FIRST IN 2 SEC")
            print(str(e))
            time.sleep(2)
            return 0,lst[0]
    return idx, obj

class ListChoices(click.Choice):
    def __init__(self, lst, names=None, text_choices=None, allow_all=True, allow_none=True, message=None, index_combos=None):
        """
        :param lst: Iterable (tuple|int) to choose from
        :param names: Names (None|Callable|list) to hint what you are choosing (else str of objs in list)
        :param allow_all: Whether or not the user can choose "All"
        :param allow_none: Whether or not the user can chose "None"
        :param combos: (combo_name, [index_list])] # NOTE THAT THIS IS INDEX NOT CHOICE!!
        returns index_of_chosen: int, lst_object(s)_chosen: list
        """
        if text_choices is None:
            text_choices = []
        self.combos = index_combos
        if message is None:
            message = "Choose one below:"
        self.min = 1
        self.max = len(lst)
        self.lst = lst
        self.first_letters = []
        self.message = message
        self.allow_all = allow_all
        self.allow_none = allow_none
        self.names = names
        self.message = message
        if index_combos is None:
            index_combos = []

        if allow_all:
            self.first_letters.append("a")
        if allow_none:
            self.first_letters.append("n")
        if self.combos:
            for c_name, _ in self.combos:
                if c_name[0].lower() in self.first_letters:
                    raise ValueError(f"Choose a different combo name (can't start with {cname[0]} because it's already taken)!")
                self.first_letters.append(c_name[0].lower())
        # self.first_letters=[tx[0].lower() for tx in self.text_choices]
        # print(f"text_choices {self.text_choices}")
        # print(f"first_letters {self.first_letters}")
        super().__init__(list(range(self.min, self.max)) + self.first_letters, case_sensitive=False)
        if not isinstance(lst, (list, tuple)):
            raise TypeError("lst argument has to be of type `list` or `tuple`")

    def name_for(self, _nopos=None, idx=None, itm=None):
        if _nopos is not None:
            raise ValueError("name_for should not get positional argument!")
        if (idx is None and itm is None) or (itm is not None and idx is not None):
            raise ValueError("ONE of `idx` and `itm` is required in name_for!")

        if idx is not None:
            itm = self.lst[idx]
        elif itm is not None:
            idx = self.lst.index(itm)
        if callable(self.names):
            name = self.names(itm)
        elif isinstance(self.names, (tuple, list)) and idx < len(self.names):
            name = self.names[idx]
        else:
            name = str(itm)
        return name

    def get_message(self):
        prompt_message = self.message
        for i, l in enumerate(self.lst, start=1):
            name=self.name_for(itm=l)
            prompt_message += f"\n{i}: {name}"
        if self.allow_all:
            prompt_message += "\n[A]ll"
        if self.allow_none:
            prompt_message += "\n[N]one"
        if self.combos:
            for c_name, idxs in self.combos:
                names=[self.name_for(idx=i) for i in idxs]
                str_names=",".join(names)
                prompt_message += f"\n[{c_name[0]}]{c_name[1:]}: {str_names}"
        prompt_message += "\nSeparated by spaces to choose multiple"
        return prompt_message

    def convert(self, raw_value, param, ctx):
        # Exact match
        values = raw_value.split(" ")
        chosen = []
        for value in values:
            if value.isdigit():
                digit = int(value)
                chosen.append(digit - 1)  # One indexed choices
            elif value.lower() in self.first_letters:
                if value.lower() in ["a", "n"]:
                    chosen.append(value.lower())
                elif self.combos:
                    combo_idxs = [idxs for cn, idxs in self.combos if cn[0].lower() == value.lower()][0]
                    print("cidx", combo_idxs)
                    chosen += combo_idxs
            else:
                self.fail(f"Invalid choice `{value}` found")
        if ("a" in chosen or "b" in chosen) and len(chosen) > 1:
            self.fail("All and None options have to be chosen alone")
        if len(chosen) == 1:
            if chosen[0] == "a":
                return (list(range(self.max)), self.lst)
            elif chosen[0] == "n":
                return ([], [])
        else:
            return (chosen, [self.lst[i] for i in chosen])

        # return super().convert(value, param, ctx)



def ask_list(lst, message=None, names=None, allow_all=True, allow_none=True, index_combos=None, failsafe=True):
    """
    lst: Iterable (tuple|int) to choose from
    names: Names (None|Callable|list) to hint what you are choosing (else str of objs in list)
    allow_all: Whether or not the user can choose "All"
    allow_none: Whether or not the user can chose "None"
    combos: (combo_name, [index_list])] # NOTE THAT THIS IS INDEX NOT CHOICE!!
    returns index_of_chosen: int, lst_object(s)_chosen: list
    """
    try:
        choices = ListChoices(lst, allow_all=allow_all, allow_none=allow_none, message=message, names=names, index_combos=index_combos)
        idx, objs = click.prompt(choices.get_message(), type=choices, confirmation_prompt=False, show_choices=False)
    except SystemExit:
        os._exit(0)
    except click.Abort:
        raise
    except Exception as e:
        if not failsafe:
            raise
        else:
            import time
            print("ERROR DOING LIST ASK! TRIGGERING ALL IN 2 SECS.")
            print(str(e))
            time.sleep(2)
            return list(range(len(lst))), lst
    return idx, objs


def test_ask_user():
    #pylint: disable=C0121
    print("All of the below should be True:")
    print(ask_user("T", "y", response="y") == True)
    print(ask_user("F", "y", response="n") == False)
    print(ask_user("T", "y", response="x") == True)
    print(ask_user("T", "y", response="") == True)
    print(ask_user("T", "n", response="y") == True)
    print(ask_user("F", "n", response="n") == False)
    print(ask_user("F", "n", response="x") == False)
    print(ask_user("F", "n", response="") == False)


def tick():
    global TICKTOCK_START
    TICKTOCK_START = time.time()


def tock(msg="Time: {}"):
    print(msg.format(time.time() - TICKTOCK_START))


def simple_csv(csv_file, newline="\n", sepchar=",", max_lines=None):
    with open(csv_file, "r") as fh:
        conts = fh.read().rstrip()
    conts = conts.split("\n")
    if max_lines is not None:
        conts = conts[:max_lines]
    conts = ",".join(conts).split(",")
    it = iter(conts)
    conts = {k: next(it) for k in it}
    return conts


def load_pickle(p_file, default=None):
    if os.path.isfile(p_file):
        fp = open(p_file, "rb")
        p_conts = pickle.load(fp)
        fp.close()
        return p_conts
    elif default is None:
        raise FileNotFoundError(p_file)
    else:
        return default


def save_pickle(data, p_file):
    with open(p_file, "wb") as fp:
        pickle.dump(data, fp)



# Slicer: Makes slice notation to slice() functions.
# e.g. if
# sl=Slicer[:1,1:]
# A=make_random_np_matrix((3,3))
# then
# A[sl] == A[:1,1:] # True


class _slicer_dummy(object):
    def __getitem__(self, val):
        return val


Slicer = _slicer_dummy()
import collections
from copy import deepcopy

def dottify_recursive_update(dct, merge_dct):
    """ Recursive dict merge. Inspired by :meth:``dict.update()``, instead of
    updating only top-level keys, dict_merge recurses down into dicts nested
    to an arbitrary depth, updating keys. The ``merge_dct`` is merged into
    ``dct``.
    :param dct: dict onto which the merge is executed
    :param merge_dct: dct merged into dct
    :return: None
    """
    # if isinstance(_merge_dct, _CFG):
    # merge_dct = dict(_merge_dct)
    # else:
    # merge_dct = _merge_dct
    for k, _ in merge_dct.items():
        if isinstance(dct, Dottify):
            key_in_dict = k in dct.__dict__
        else:
            key_in_dict = k in dct
        if key_in_dict and isinstance(dct[k], (dict, Dottify)) and isinstance(merge_dct[k], (collections.Mapping, Dottify)):
            dottify_recursive_update(dct[k], merge_dct[k])
        else:
            # if key_in_dict:
            # print(f"Endval: ({type(v)}) {k}:{merge_dct[k]}: {isinstance(merge_dct[k],_CFG)}", "")
            # print((k in dct),isinstance(dct[k], (dict, _CFG)),isinstance(merge_dct[k], (collections.Mapping, _CFG, util.Dottify)))
            dct[k] = merge_dct[k]

class Dottify(object):
    """docstring for SPLITS."""

    def __init__(self, from_dict=None, **kwargs):
        if from_dict is None:
            from_dict = {}
        self.__dict__ = {}
        self.__dict__.update(from_dict)
        self.__dict__.update(kwargs)
        if hasattr(self, "setup"):
            self.setup()
        self._set_attr_from_dict()
    # @property
    # def classvars(self):
    #     cv_dict={}
    #     for k,v in self.__dict__.items():
    #         if not k.startswith("_"):
    #             cv_dict[k]=v
    #     return cv_dict
    def __getattr__(self, name):
        if hasattr(self.__dict__, name):
            return getattr(self.__dict__, name)
        # super(Dottify, self).__getattr__(name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def _set_attr_from_dict(self):
        for k, v in self.__dict__.items():
            setattr(self, k, v)

    def merge_cp(self, dottify_or_dict, *dottifies_or_dicts, copy=True):
        if dottifies_or_dicts:
            dottifies_or_dicts=list(dottifies_or_dicts)
            dottifies_or_dicts=[*ensure_list(dottify_or_dict), *dottifies_or_dicts]
        else:
            dottifies_or_dicts=ensure_list(dottify_or_dict)
        if copy:
            template=deepcopy(self)
        else:
            template=self
        for c in dottifies_or_dicts:
            dottify_recursive_update(template, c)
        # print(template)
        return template
        # return template

    # @property
    # def __dict__(self):
    #     return self.__dict__
    # @__dict__.setter
    # def __dict__(self, val):
    #     self.__dict__=val

    def hide_from_dict(self,key):
        del self.__dict__[key]

    def override(self, **kwargs):
        self.__dict__.update(kwargs)
        self._set_attr_from_dict()
    def update(self, udict):
        if isinstance(udict, Dottify):
            self.__dict__.update(udict.__dict__)
        elif isinstance(udict, dict):
            self.__dict__.update(udict)
        else:
            raise TypeError("Cannot update Dottify with type {type(udict)}")
        self._set_attr_from_dict()
    def __len__(self):
        return self.__dict__.__len__()
    def items(self):
        return self.__dict__.items()
    def keys(self):
        return self.__dict__.keys()
    def values(self):
        return self.__dict__.values()
    def _iterator(self):
        for k, v in self.__dict__.items():
            yield k, v

    def __getitem__(self, itm):
        return self.__dict__[itm]
    def __setitem__(self, itm,val):
        self.__dict__[itm]=val
        # setattr(self,itm, val)
        # return self.__dict__[itm]=val
        pass

    def __iter__(self):
        return self._iterator()
    # def __str__(self):
        # return str(self.__dict__)

    def __repr__(self):
        return str(self.__dict__)



fd = sys.stdin.fileno()
termios_old_settings = termios.tcgetattr(fd)
# termios_old_settings[3]=0b100000000000000000010111001111
termios_new_settings = termios.tcgetattr(fd)
termios_new_settings[3] = termios_new_settings[3] & ~(termios.ICANON | termios.ECHO) | (termios.NOFLSH)# lflags

def read_stdin(vtime=0, vmin=0):
    termios_new_settings[6][termios.VMIN] = vmin  # cc
    termios_new_settings[6][termios.VTIME] = vtime*10 # cc
    termios.tcsetattr(fd, termios.TCSANOW, termios_new_settings)
    ch_set = []
    ch = os.read(sys.stdin.fileno(), 1)
    while ch != None and len(ch) > 0:
        ch_set.append(chr(ch[0]))
        ch = os.read(sys.stdin.fileno(), 1)
    termios.tcflush(sys.stdin, termios.TCIFLUSH)
    # sys.stdin.flush()
    termios.tcsetattr(fd, termios.TCSANOW, termios_old_settings)
    return "".join(ch_set)
atexit.register(lambda: termios.tcsetattr(fd, termios.TCSADRAIN, termios_old_settings))

def call_if_lambda(f):
    """
    Calls f if f is a lambda function.
    From https://stackoverflow.com/a/3655857/997253
    """
    LMBD = lambda:0
    islambda=isinstance(f, type(LMBD)) and f.__name__ == LMBD.__name__
    return f() if islambda else f

class _UNSPECIFIED():
    pass

def get_valv(key, default=_UNSPECIFIED):
    if key in sys.argv:
        return sys.argv[sys.argv.index(key)+1]
    elif default is _UNSPECIFIED:
        raise KeyError(f"The key {key} is not found in argument vector: (argv)!")
    return default


if __name__ == '__main__':
    # res=ask_choices("Message?", ["list", "of", "opts"])
    res=ask_choose_one(["list", "of", "opts"],"Message?")
    print("ANS")
    print(res)
    # class Tst(object):
    #     def __init__(self,arg):
    #         self.__dict__['arg']=arg
    #
    # t=Tst("hey")
    # print(t.arg)
#     print(termios_new_settings)
#     try:
#         sys.stdin.flush()
#         sys.stdout.flush()
#         while True:
#             print("\nREAD\n")
#             st=read_stdin()
#             print("-"*7)
#             print("OUT:", st)
#             print("SLEEP")
#             print("-"*7)
#             # sys.stdout.flush()
#             time.sleep(2)
#     finally:
#         termios.tcsetattr(fd, termios.TCSADRAIN, termios_old_settings)
#         pass
    # class teststuff(Dottify):
    #     some_thing="foo"
    # t=teststuff(some_thing="hello")
    # print(t.some_thing)
    # t.some_thing="world"
    # print(t.some_thing)
    # for k,v in t.items():
    #     print(f"{k}={v}")

class _MARKER():
    def __init__(self,mark, value=None):
        self.mark=mark
        self.value=value
SMARK=_MARKER
MARK=_MARKER
# class _SYNTAX_MARKER():
#     def __eq__(self,other):
#         print(other.__class__, self.__class__)
#         # print(other.__parent__)
#         return (isinstance(other, self.__class__) and other.mark == self.mark)
#
# def SYNTAX_MARKER(klass):
#     klass=klass()
#     cls = klass.__class__
#     klass.__class__ = type(cls.__name__+"_MARKER", (cls, _SYNTAX_MARKER), {})
#     return klass
class ArgumentError(BaseException):
    pass
def save_pickle_cache(file, value):
    pickle_path=relpath(f"../../../data/pickles/{file}.p")
    with open(pickle_path, 'wb') as f:
        return pickle.dump(value, f)

def try_load_pickle_cache(file, default=False):
    pickle_path=relpath(f"../../../data/pickles/{file}.p")
    if not os.path.isfile(pickle_path):
        return default
    elif "--clearcache" in sys.argv or "--recache" in sys.argv:
        if ask_user(f"Delete pickle file {pickle_path}?", default=False, y_args=["--overwrite"]):
            os.remove(pickle_path)
        return default
    elif "--nocache" in sys.argv:
        return default
    else:
        with open(pickle_path, 'rb') as f:
            return pickle.load(f)


def pickle_cache(file, default=False, save_cache=True, autopath=True, cb_kwargs=None, uncacheable=True):
    cb_kwargs=cb_kwargs or {}
    uncacheable=("--force-clearcache" in sys.argv) or uncacheable
    if autopath:
        # pickle_path=relpath(f"../../pickles/{file}.p")
        pickle_path=relpath(f"../../../data/pickles/{file}.p")
    else:
        pickle_path=file
        file=os.path.basename(file)
    if "--clearcache" in sys.argv and os.path.isfile(pickle_path) and ask_user(f"Delete pickle for {file}?", default=True, y_args=["--overwrite"]):
        os.remove(pickle_path)

    def do_return():
        val=default
        cprints.color_print(f"No pickle found at {pickle_path}.")
        # if not ask_user("Continue?", default=True):
        #     raise click.Abort("User chose to not continue.")
        if callable(default):
            val=val(**cb_kwargs)
        if save_cache:
            print("Dumping pickle")
            with open(pickle_path, "wb") as f:
                pickle.dump(val, f)
        return val
    if "--nocache" in sys.argv or ("--clearcache" in sys.argv and uncacheable):
        if callable(default):
            return default(**cb_kwargs)
        else:
            return default
    else:
        cache_file_exists=os.path.isfile(pickle_path)
        if (cache_file_exists and not uncacheable) or (cache_file_exists and ("--recache" not in sys.argv)):
            cprints.color_print(f"Loading pickle cache {file} from {pickle_path}", style="warning")
            with open(pickle_path, 'rb') as f:
                return pickle.load(f)
        else:
            return do_return()
