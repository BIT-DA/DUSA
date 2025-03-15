from imagecorruptions import corrupt, get_corruption_names
from argparse import ArgumentParser
from functools import partial
import os
import os.path as osp
from PIL import Image
import numpy as np


class RecursiveCounter:
    def __init__(self):
        self._value = 0

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, _value):
        self._value = _value

    def __iadd__(self, other: int):
        self._value += other
        return self


def main():
    parser = ArgumentParser("Build ADE20k-C form ADE20k val set!")
    parser.add_argument("-c",
                        type=str,
                        default="",
                        help="corruption type")
    parser.add_argument('-s',
                        type=int,
                        default=-1,
                        help="severity of the corruption, defaults 5")

    parser.add_argument('--ade20k-val-img-dir',
                        type=str,
                        default="",
                        help="dir to the ade20k val dir")

    # f"{corruptions-out-dir}/{c}/{s}/xxx"
    parser.add_argument('--corruptions-out-dir',
                        type=str,
                        default="",
                        help="corruptions output dir")

    arg = parser.parse_args()

    assert arg.c != "" and arg.s in [1, 2, 3, 4, 5], "please set corruption type of severity!"
    assert arg.ade20k_val_img_dir != "" and arg.corruptions_out_dir != "", "please set img dir, out dir!"

    corruption_func = partial(corrupt, severity=arg.s, corruption_name=arg.c)
    src_dir = arg.ade20k_val_img_dir
    des_dir = osp.join(arg.corruptions_out_dir, arg.c, str(arg.s))

    counter = RecursiveCounter()
    error_info = []
    process_dir(corruption_func, src_dir, des_dir, counter, error_info)
    print("All Error info\n" + "\n".join(error_info))


def process_dir(corruption_func, src_dir, des_dir, counter, error_info):
    dir_list = os.listdir(src_dir)
    for name in dir_list:
        if name == "training":
            continue
        src_sub_path = osp.join(src_dir, name)
        des_sub_path = osp.join(des_dir, name)

        if osp.isdir(src_sub_path):
            os.makedirs(des_sub_path, exist_ok=True)
            process_dir(corruption_func, src_sub_path, des_sub_path, counter, error_info)
        elif osp.isfile(src_sub_path):
            process_img(corruption_func, src_sub_path, des_sub_path, counter, error_info)
        else:
            continue


def process_img(corruption_func, src_sub_path, des_sub_path, counter, error_info):
    # if not src_sub_path.endswith(".png"):
    #     # not is image file
    #     return
    try:
        image = Image.open(src_sub_path)
        image = np.array(image)
        corrupted_image = corruption_func(image)
        corrupted_image = Image.fromarray(corrupted_image)
        corrupted_image.save(des_sub_path)
        counter += 1
        print(f"Finished {counter.value}: {src_sub_path} -> {des_sub_path}")

    except Exception as e:
        error_info.append(f"Error when processing {src_sub_path}, error info: " + str(e))
    finally:
        return


if __name__ == "__main__":
    main()

