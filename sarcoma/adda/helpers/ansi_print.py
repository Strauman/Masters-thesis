import sys
swrite=sys.stdout.write
import asyncio

def ansi_cmd(string):
    return "\033[{}".format(string)


def ansi_tx_color(c_num="255"):
    return ansi_cmd("38;5;{}m".format(c_num))


def ansi_bg_color(c_num="019"):
    return ansi_cmd("48;5;{}m".format(c_num))

import shutil
screen_columns, screen_rows = shutil.get_terminal_size(fallback=(80, 24))
ansi = type('ansiseq', (object,), {})()
ansi.curup = ansi_cmd("F")  # Cursor up
ansi.clearline = ansi_cmd("K")  # Clear to end of line
ansi.clearscreen = ansi_cmd("1J")  # Erase display
ansi.previous_line = ansi_cmd("F")
ansi.hide_cursor = ansi_cmd("?25l")
ansi.show_cursor = ansi_cmd("?25h")
ansi.alternative_screen_buffer = ansi_cmd("?1049h")
ansi.topleft = ansi_cmd("1;1H")
ansi.end_color = ansi_cmd("0m")
ansi.restore_cursor = ansi_cmd("u")
ansi.save_cursor = ansi_cmd("s")
def ansip(ans):
    swrite(ans)

def pbar_print(msg, lines_above=1):
    swrite(ansi.save_cursor)
    swrite("".join([ansi.previous_line for _ in range(lines_above)]))
    swrite(ansi.clearline)
    swrite(msg)
    swrite(ansi.restore_cursor)
    sys.stdout.flush()

def goto(x,y):
    return ansi_cmd("{};{}H".format(x,y))

def write_top_right(text, r_offset=0):
    sys.stdout.write(ansi.save_cursor)
    screen_columns, screen_rows = shutil.get_terminal_size(fallback=(80, 24))
    x=screen_columns-len(str(text))-2
    y=0+r_offset
    sys.stdout.write(goto(y,x))
    sys.stdout.write(str(text))
    sys.stdout.write(ansi.restore_cursor)
    sys.stdout.flush()
import time
def clearscreen():
    sys.stdout.write(ansi.clearscreen)
    sys.stdout.flush()

from collections import OrderedDict

NOTIFICATIONS=OrderedDict()
def print_notifications():
    global NOTIFICATIONS
    for i,n in enumerate(NOTIFICATIONS.values()):
        write_top_right(n,i+1)


from multiprocessing import Pool
import threading
IS_EXITING=False
def notifier_loop(interval=1):
    global IS_EXITING
    try:
        while True:
            print_notifications()
            time.sleep(interval)
            if IS_EXITING:
                return
    except SystemExit:
        return
    #
def say_hello():
    sys.stdout.write("HELLO")
    sys.stdout.flush()

# # thr=T(target=f, args=(), kwargs={})
# class StoppableThread(threading.Thread):
#     """Thread class with a stop() method. The thread itself has to check
#     regularly for the stopped() condition."""
#
#     def __init__(self, *args, **kwargs):
#         super(StoppableThread, self).__init__(*args, **kwargs)
#         self._stop_event = threading.Event()
#
#     def stop(self):
#         self._stop_event.set()
#
#     def stopped(self):
#         return self._stop_event.is_set()


def run_notifier(interval=1):
    thr = threading.Thread(target=notifier_loop, args=(), kwargs=dict(interval=0.5))
    thr.daemon = True
    # pool = Pool(processes=1)
    # pool.apply_async(f, "x", say_hello)
    thr.start()

import atexit
# def tell_exit():
#     global thr
#     print("Telling thr to stop!")
#     if thr is not None:
#         thr.stop()
#
# # atexit.register(tell_exit)


if __name__ == '__main__':
#     i=0
#     # save_cursor="\u001b[{s}"
#     # restore_cursor="\u001b[{u}"
#
#     while True:
#         msg="Hello {}".format(i)
#         i+=1
#         write_top_right("Hello world!",10)
#         time.sleep(1)
    NOTIFICATIONS["number"]=0
    run_notifier()
    i=0
    while True:
        i+=1
        NOTIFICATIONS["number"]+=1
        print("Num", NOTIFICATIONS["number"])
        time.sleep(0.5)
        if i>=5:
            # print("Attempting stop")
            # thr.stop()
            # break
            raise Exception("Some exception")
