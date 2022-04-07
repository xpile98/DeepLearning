import inspect
import os


def print_shapes(*args):
    out_string = "Line %-5d:\t"% inspect.getlineno(inspect.getouterframes(inspect.currentframe())[-1][0])
    for i in args:
        out_string = out_string + ("%-25s" % str(list(i.shape)))
    print(out_string)

def load_dir_data(base_path, size=(224,224), channel=3, shuffle=True, seed=None):
    base_dir = os.listdir(base_path)
    n_classes = len(base_dir)

    if base_path[-1] != '/':
        base_path = base_path + '/'
