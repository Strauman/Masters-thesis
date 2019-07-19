import re
import pout


class Variabled(str):
    def format_vars(self, variables, var_sym="§"):
        ret = self
        for key, val in variables.items():
            ret = re.sub(r"§{}(\b)".format(key), val + "\1", ret)
        # return Variabled(ret)
        return ret


def builder(f):
    def _wrapper(self, *args, **kwargs):
        if self._built:
            raise AttributeError("{f.__name__} can't be called after TeXEnv is built")


class TeXEnv(object):
    def __init__(self, name, opt_args=None, req_args=None):
        self.name = name
        self.opt_args = []
        self.req_args = []
        self._built = False
        self.body = ""
        self.show_empty_oarg=False
        self.show_empty_rarg=False

    def add_opt(self, arg):
        self.opt_args.append(arg)

    def add_req(self, arg):
        self.req_args.append(arg)

    def finish(self):
        oargs=""
        if self.opt_args or self.show_empty_oarg:
            oargs=f"[{','.join(self.opt_args)}]"
        if self.req_args or self.show_empty_rarg:
            rargs=f"{{{','.join(self.req_args)}}}"
        variables = dict(
            env_name=self.name,
            oargs=oargs,
            rargs=rargs,
            BODY=self.body
        )
        out_string = Variabled(r'''
        \begin{§env_name}§oargs§rargs
        §BODY
        \end{§env_name}
        ''').format_vars(variables)
        self._built = True
        return out_string


# class TikzPicture(object):
#     """docstring for TikzPicture."""
#
#     def __init__(self, arg):
#         super(TikzPicture, self).__init__()
#         self.arg = arg
#
#

def _tiks_boxplot(args):
    tikzpic=TeXEnv("tikzpicture")
    axis=TeXEnv("axis")
    default_args=dict(
        median=1,
        "upper quartile"=1.2,
        "lower quartile"=0.4,
        "upper whisker"=1.5,
        "lower whisker"=0.2
    )
    axis.add_opt('boxplot/draw direction=y')
    axis.body=Variabled(r'''
        \addplot+[
        boxplot prepared={
        §args
        },
        ] coordinates {};
    ''').format()

if __name__ == '__main__':
    # v=Variabled("§arg/§argv").format_vars(dict(arg="ARGUMENT", argv="ARGV"))
    # print(v)
    # doc = TeXEnv("document")
    # doc.body = "HELLO WORLD"
    # doc.add_opt("optional=argument")
    # doc.add_req("requiredArgument")
    # print(doc.finish())
