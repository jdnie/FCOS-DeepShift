'''
从log日志中抽取模式信息并用图表展示
'''

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import re
import datetime
import argparse


def re_match_float(log, float_perfix="loss: "):
    re_float = r"\d+.\d+"
    re_float_perfix = float_perfix + re_float
    out = []
    try:
        for i in re.findall(re_float_perfix, log):
            out += re.findall(re_float, i)
    except:
        print("Analsis failed at line: {}".format(log))
    return out


def draw_graph_from_log(log_file, float_perfix, jpg_path, ignore_num=0, is_show=False):
    with open(log_file, 'r') as f:
        out = []
        for l in f.readlines():
            out += re_match_float(l, float_perfix=float_perfix)
        data = [float(i) for i in out]
        # print(data)
        plt.figure()
        plt.plot(data[ignore_num:])
        plt.title(float_perfix)
        plt.savefig(jpg_path)
        if is_show:
            plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_path', type=str, help='log path',
                        default="./")
    parser.add_argument('--float_perfix', type=str, help='float perfix',
                        default="loss: ")
    parser.add_argument('--jpg_path', type=str, help='jpg path',
                        default=None)
    args = parser.parse_args()

    ignore_num = 10
    is_show = False
    draw_graph_from_log(args.log_path,
                        float_perfix=args.float_perfix,
                        jpg_path=args.jpg_path,
                        ignore_num=ignore_num,
                        is_show=is_show)


if __name__ == "__main__":
    main()
