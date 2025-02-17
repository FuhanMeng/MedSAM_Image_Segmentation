# 绘制流程图
from pycallgraph import PyCallGraph
from pycallgraph.output import GraphvizOutput
import os
import train_one_gpu as fun_test


def main():
    graphviz = GraphvizOutput()
    # 可视化图片保存路径
    project_path = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(project_path, 'output', 'basicdemo.png')
    graphviz.output_file = output_path
    # graphviz.output_file = r'C:\Users\user\Desktop\test_code\devin-tools\basicdemo.png'

    with PyCallGraph(output=graphviz):
        fun_test.main()
if __name__ == '__main__':
    main()

