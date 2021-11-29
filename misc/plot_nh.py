# @File :plot_nh.py
# @Time :2021/11/21
# @Desc :
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

plt.rc('font', family="Times New Roman")
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['patch.linewidth'] = 1.2
plt.rcParams['axes.linewidth'] = 1.2
SAVE_DIR = 'eval/misc/plot'


def plot_param():
    acc_knowit_qa = [76.83, 77.00, 77.10, 76.95, 76.91]
    x = np.arange(0, len(acc_knowit_qa))
    x_index = [1, 4, 8, 16, 32]
    fig = plt.figure(figsize=(8, 6))
    plt.xticks(fontproperties='Times New Roman', size=24)
    plt.yticks(fontproperties='Times New Roman', size=24)
    plt.plot(x, acc_knowit_qa, color='c', linewidth=2, markersize=10,
             marker='D', linestyle='-', label="KnowIT VQA")
    # 标注数值
    for xi, yi in zip(x, acc_knowit_qa):
        plt.text(xi, yi + 0.3, yi, fontsize=20,
                 verticalalignment='top', horizontalalignment='center')
    plt.ylim((75.00, 78.80))
    my_y_ticks = np.arange(75.00, 78.80, 0.50)
    plt.yticks(my_y_ticks, fontproperties='Times New Roman', size=22)
    plt.xticks(x, x_index, fontproperties='Times New Roman', size=22)
    plt.legend(
        fontsize=22,
        loc="best",
        # prop={"family": "Times New Roman", 'size': 24},
        # loc="lower center",
        # bbox_to_anchor=(0.25, 0.02)
    )
    plt.xlabel("$N_h$",
               fontproperties='Times New Roman',
               fontsize=24)
    plt.ylabel("Accuracy (%)",
               fontproperties='Times New Roman',
               fontsize=24)
    plt.grid(True)
    # plt.show()

    plt.savefig(
        os.path.join(SAVE_DIR, f'knowit_vqa.svg'),
        bbox_inches='tight',
        dpi=1200,
        pad_inches=0.0,
        format='svg'
    )
    plt.show()
    
    
    acc_msrvtt_qa = [57.83, 58.61, 59.30, 59.44, 59.09]

    x = np.arange(0, len(acc_msrvtt_qa))
    x_index = [1, 4, 8, 16, 32]
    fig = plt.figure(figsize=(8, 6))
    plt.xticks(fontproperties='Times New Roman', size=24)
    plt.yticks(fontproperties='Times New Roman', size=24)
    plt.plot(x, acc_msrvtt_qa, color='b', linewidth=2, markersize=10,
             marker='D', linestyle='-', label="MSRVTT-QA")
    
    # 标注数值
    for xi, yi in zip(x, acc_msrvtt_qa):
        plt.text(xi, yi+0.3, yi, fontsize=20,
                 verticalalignment='top', horizontalalignment='center')
    
    plt.ylim((57.00, 60.80))
    my_y_ticks = np.arange(57.00, 60.80, 0.50)
    plt.yticks(my_y_ticks, fontproperties='Times New Roman', size=22)
    plt.xticks(x, x_index, fontproperties='Times New Roman', size=22)
    plt.legend(
        fontsize=22,
        loc="best",
        # prop={"family": "Times New Roman", 'size': 24},
        # loc="lower center",
        # bbox_to_anchor=(0.25, 0.02)
    )
    plt.xlabel("$N_h$",
               fontproperties='Times New Roman',
               fontsize=24)
    plt.ylabel("Accuracy (%)",
               fontproperties='Times New Roman',
               fontsize=24)
    plt.grid(True)
    # plt.show()

    plt.savefig(
        os.path.join(SAVE_DIR, f'msrvtt_qa.svg'),
        bbox_inches='tight',
        dpi=1200,
        pad_inches=0.0,
        format='svg'
    )
    plt.show()


def main():
    methods = ["LiVLR-V(Ours)", "MASN", "DualVGR", "GRA", "ST-VQA",
               "HCR", "HME", "Co-men", "ClipBERT", "HGA", "VQA-T"]
    params = [10.7, 28.2, 34.1, 35.4, 39.0,
              43.7, 48.3, 69.5, 113.5, 121.4, 156.5]  # M
    acc_all = [40.6, 35.2, 35.5, 32.5, 30.9,
               35.6, 33.0, 32.0, 37.4, 35.5, 41.5]

    colors = matplotlib.cm.rainbow(np.linspace(0, 1, len(acc_all)))
    # colors = matplotlib.cm.jet(np.linspace(0, 1, len(acc_all)))
    fig = plt.figure(figsize=(8, 6))
    ax = plt.gca()
    ax.tick_params(axis='both', width=5)
    plt.xticks(fontproperties='Times New Roman', size=22)
    plt.yticks(fontproperties='Times New Roman', size=22)
    # plt.scatter(params, acc_all, s=100, c=colors, cmap="jet", )

    for i, (p, a) in enumerate(zip(params, acc_all)):
        plt.scatter(p, a, s=200, color=colors[i], label=methods[i])
        if methods[i] == "MASN":
            plt.text(p, a - 1.1, f"{p}M", fontsize=20,
                     verticalalignment='bottom', horizontalalignment='center')
        elif methods[i] == "HCR":
            plt.text(p + 22, a, f"{p}M", fontsize=20,
                     verticalalignment='center', horizontalalignment='right')
        elif methods[i] == "ClipBERT":
            plt.text(p + 25, a, f"{p}M", fontsize=20,
                     verticalalignment='center', horizontalalignment='right')
        elif methods[i] == "LiVLR-V(Ours)":
            plt.text(p+2, a + 1.0, f"{p}M", fontsize=20,
                     verticalalignment='top', horizontalalignment='center')
        elif methods[i] == "VQA-T":
            plt.text(p-5, a + 1.0, f"{p}M", fontsize=20,
                     verticalalignment='top', horizontalalignment='center')
        else:
            plt.text(p, a + 1.0, f"{p}M", fontsize=20,
                     verticalalignment='top', horizontalalignment='center')
    
    plt.legend(
        fontsize=20,
        labelspacing=0.1,
        columnspacing=0.1,
        handletextpad=0.02,
        # borderpad=0.1,
        handlelength=1.2,
        frameon=True,
        # bbox_to_anchor=(1.02, 1),
        # shadow=True,
        edgecolor="black",
        # facecolor="black",
        # framealpha=0.2,
        loc="upper center",
        ncol=2,
        borderaxespad=0.2
    )

    plt.ylim((30.0, 44.5))
    my_y_ticks = np.arange(30.0, 44.5, 2)
    plt.yticks(my_y_ticks, fontproperties='Times New Roman', size=22)
    plt.xlabel("Model parameters (M)",
               fontproperties='Times New Roman',
               fontsize=24)
    plt.ylabel("Accuracy (%)",
               fontproperties='Times New Roman',
               fontsize=24)
    plt.grid(True)
    # plt.show()

    plt.savefig(
        os.path.join(SAVE_DIR, f'pa.svg'),
        bbox_inches='tight',
        dpi=1200,
        pad_inches=0.0,
        format='svg'
    )
    plt.show()


if __name__ == "__main__":
    main()
    # plot_param()
