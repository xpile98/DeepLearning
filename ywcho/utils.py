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

def keras_model_info_save(model, name):
    name = name[(name[:-3].rfind("/")) + 1 : -3]

    # 모델 구조 (그림) 저장
    from tensorflow import keras
    keras.utils.plot_model(model, show_shapes=True, to_file=name + '.png', dpi=300)

    # 모델 파일 저장 (.h5)
    model.save(name + '.h5')  # creates a HDF5 file 'my_model.h5'
    del model  # deletes the existing model
    model = keras.models.load_model(name + '.h5')

    # 모델 summary 저장
    from contextlib import redirect_stdout
    with open(name + '.txt', 'w') as f:
        with redirect_stdout(f):
            model.summary()