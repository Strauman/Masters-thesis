import util
#------ Warnings and messages ------#
#------ Colorized output ------#
# For more colors: https://askubuntu.com/questions/558280/changing-colour-of-text-and-background-of-terminal
#------ Font colors ------#
format_color = dict(OKBLUE='\033[94m',
                    WHITE='\033[38;5;255m',
                    BLACK='\033[38;5;16m',
                    OKGREEN='\033[38;5;46m',
                    FATAL='\033[38;5;196m',
                    WARNING='\033[38;5;190m',
                    FAIL='\033[91m',
                    NOTICE='\033[38;5;136m',
                    #------ Backgrounds ------#
                    BLUEBG='\033[48;5;021m',
                    OKGREENBG='\033[48;5;46m',
                    NOTICEBG='\033[48;5;136m',
                    FATALBG='\033[48;5;196m',
                    GREYBG='\033[48;5;250m',
                    CYANBG='\033[48;5;051m',
                    WARNINGBG='\033[48;5;208m',
                    INPUTBG='\033[48;5;194m',
                    #------ Font styles ------#
                    BOLD='\033[1m',
                    UNDERLINE='\033[4m',
                    #------ End color ------#
                    ENDC='\033[0m')
cprint_color_styles = {
    'danger': '{FATALBG}{WHITE}',
    'error': '{FATALBG}{WHITE}',
    'success': '{OKGREENBG}{BLACK}',
    'warning': '{WARNINGBG}{BLACK}',
    'notice': '{BLUEBG}{WHITE}'
}

#--- STYLES ---#


def add_format_colors(**kwargs):
    for key, fmat in kwargs.items():
        if key in format_color:
            raise ValueError(f"Color format {key} already exists")
        format_color[key] = fmat.format(**format_color)


add_format_colors(
    S_DANGER='{FATALBG}{WHITE}',
    S_ERROR='{FATALBG}{WHITE}',
    S_SUCCESS='{OKGREENBG}{BLACK}',
    S_WARNING='{WARNINGBG}{BLACK}',
    S_NOTICE='{BLUEBG}{WHITE}'
)


def print_cstyles():
    for k, prefix in cprint_color_styles.items():
        print("{}{}{}".format(prefix, k, format_color['ENDC']))


def color_print(*r_msgs, style=None, as_str=False, plain=False, **format_keys):
    # format_keys.update(format_color)
    style_prefix = style_suffix = ""
    join_char="\t"
    r_msgs = list(r_msgs)
    msgs=[]+r_msgs
    if plain:
        if as_str:
            return "\t".join(r_msgs)
        print(*r_msgs)
        return True
    style = style or "notice"
    if style.lower() in cprint_color_styles.keys():
        style_prefix = "{}".format(cprint_color_styles[style.lower()])
        style_suffix = "{ENDC}"
        # Add prefix and suffix to all
        msgs = ["{}{}{}".format(style_prefix, msg, style_suffix) for msg in msgs]
    else:
        print("Unknown color style `{}`. Try one of these:".format(style))
        print_cstyles()

    msgs = [msg.format(**dict(**format_color, **format_keys)) for msg in msgs]
    if as_str:
        if len(msgs) == 1:
            return msgs[0]
        else:
            return msgs.join(join_char)
    else:
        print(*msgs)
    return True


def warning_print(*args, fatal=False, as_str=False):
    if len(args) == 2:
        hdr = args[0]
        message = args[1]
    elif len(args) == 1:
        hdr = args[0]
        message = None
    else:
        hdr = message = None

    if hdr is None and message is None:
        return color_print("{WARNING}{BOLD}Warning missing text...{ENDC}", as_str=as_str)
    else:
        if hdr is not None:
            hdr_str = ""
            if fatal:
                hdr_str = "{FATALBG}{WHITE}{hdr}{ENDC}"
            else:
                hdr_str = "{WARNINGBG}{BLACK}{hdr}{ENDC}"

            color_print(hdr_str, hdr=hdr, as_str=as_str)
        if message is not None:
            msg_str = ""
            if fatal:
                msg_str = "{FATAL}{message}{ENDC}"
            else:
                msg_str = "{WARNING}{BLACK}{message}{ENDC}"

            color_print(msg_str, message=message, as_str=as_str)
    fname, fpath, line = util.caller_id(1)
    color_print("in {}:{}".format(fname, line))
# def assert_warn(cond,msg):
#     if cond:
#         warning_print(msg)


def print_all_colors():
    import sys
    for i in range(16, 256):
        j = str(i).zfill(3)
        sys.stdout.write("\033[38;5;0m\033[48;5;{}m{}".format(i, j))
        sys.stdout.write("\033[0m")
        # sys.stdout.write(' ')
        if (i - 15) % 6 == 0:
            sys.stdout.write('\n')
        else:
            sys.stdout.write(' ')
        sys.stdout.flush()


if __name__ == '__main__':
    # print_all_colors()
    msg = "{WARNINGBG}{BLACK}WBBLACK"
    color_print(f"NOTICE{{ENDC}}{{S_WARNING}}{msg}", style="Notice")
