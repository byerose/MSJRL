import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import datetime
import plotly.express as px
from sklearn.decomposition import PCA
from collections import defaultdict
from adjustText import adjust_text

fontsize1 = 12
fontsize2 = 12
fontsize3 = 12


def visualize_act_with_t_sne(
    model_name,
    tensors_flat,
    labels,
    dataset_name,
    num_samples,
    layer,
    shot,
    reward,
    perplexity=30,
    scores=None,
    save=False,
):
    """
    使用 t-SNE 可视化模型的残差流的权重。

    参数:
    - model_name: 模型名称，用于标题和保存路径。
    - tensors_flat: 需要可视化的张量，通常是模型层的输出。
    - labels: 每个张量对应的标签，用于区分类别。
    - dataset_name: 数据集名称，用于保存路径。
    - num_samples: 使用的样本数量。
    - layer: 当前可视化的模型层数。
    - perplexity: t-SNE 参数，影响聚类的平滑性，默认为 30。
    - scores: 可选参数，可能用于其他分析（当前未使用）。
    - save: 是否保存生成的图像，默认为 False。
    """

    # 打印提示信息，表明正在进行 t-SNE 可视化
    print("t-SNE visualization of the weights of the residual stream of the model")

    # 初始化 t-SNE 模型，降维到2个组件（n_components=2）以便在平面中可视化
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=40)

    # 打印提示信息，显示正在拟合 t-SNE，并输出输入张量的内容
    # print("Fitting t-SNE...", [tensor for tensor in tensors_flat])

    # 使用 t-SNE 对张量进行降维
    tensors_tsne = tsne.fit_transform(tensors_flat)

    # 打印降维后张量的形状，确保符合预期
    # print("Shape of tensors_tsne:", tensors_tsne.shape)

    # 确保标签数量与降维后的样本数量匹配，如果不匹配则抛出断言错误
    assert (
        len(labels) == tensors_tsne.shape[0]
    ), "The number of labels must match the number of samples in tensors_tsne."

    # 创建一个新的图形窗口，设置大小为10x8英寸
    plt.figure(figsize=(4, 4))

    # 获取唯一标签，用于区分不同类别
    unique_labels = np.unique(labels)
    # print(unique_labels)
    # 创建一个颜色映射，使用 'tab20' 或 'Dark2' 色彩表生成和类别数量相同的颜色
    colors = cm.get_cmap("hsv", len(unique_labels))
    color_map = {"safe": ["#00008B", "#9ACD32"], "unsafe": ["#006400", "#8B0000"]}
    marker_map = {"safe": "o", "unsafe": "s"}
    # print(color_map)
    # 存储文本对象以便后续调整
    texts = []

    # 遍历每个唯一标签，分别绘制对应类别的点
    for label in unique_labels:
        # 找到该标签对应的样本索引
        indices = np.where(labels == label)[0]
        # print(indices)

        # 绘制该标签的点，使用对应的颜色，透明度为0.6，点的大小为100
        plt.scatter(
            tensors_tsne[indices[indices < 20], 0],
            tensors_tsne[indices[indices < 20], 1],
            color=color_map[label][0],
            alpha=0.7,
            s=100,
            marker=marker_map[label],
            label=label,
        )
        plt.scatter(
            tensors_tsne[indices[indices > 19], 0],
            tensors_tsne[indices[indices > 19], 1],
            color=color_map[label][1],
            alpha=0.7,
            marker=marker_map[label],
            s=80,
            label=label,
        )

        # 计算该标签点的平均位置，以便放置注释
        if label == "safe":
            mean_x = np.mean(tensors_tsne[indices[indices < 20], 0])
            mean_y = np.mean(tensors_tsne[indices[indices < 20], 1])
        if label == "unsafe":
            mean_x = np.mean(tensors_tsne[indices[indices > 19], 0])
            mean_y = np.mean(tensors_tsne[indices[indices > 19], 1])

        # # 为该类别添加文本注释，注释位置为该类别点的平均位置
        # text = plt.text(
        #     mean_x,
        #     mean_y,
        #     str(label),
        #     fontsize=fontsize2,
        #     ha="center",
        #     va="center",
        #     bbox=dict(facecolor="white", alpha=0.5, edgecolor="none", pad=1),
        # )
        # # 将文本对象存储起来，方便后续调整位置
        # texts.append(text)

    # 调整文本位置以避免重叠，特别是沿着 y 轴移动文本，使用红色箭头表示移动路径
    adjust_text(
        texts,
        only_move={"points": "y", "text": "y"},
        arrowprops=dict(arrowstyle="->", color="r", lw=0.5),
    )

    # 设置图像的标题，包括模型名称、层数、样本数量和 perplexity 参数
    # plt.title(f"t-SNE Visualization of Residual Stream Weights, model={model_name}, layer={layer}, num_samples={num_samples}, perplexity={perplexity}")
    plt.legend(
        ["vanilla_safe", "jailbreak_safe", "vanilla_harmful", "jailbreak_harmful"]
    )
    # 设置 x 轴和 y 轴的标签，使用较大的字体大小
    plt.xlabel(
        "T-SNE Component 1", fontsize=fontsize2
    )  # 使用较大的字体大小设置 X 轴标签
    plt.ylabel(
        "T-SNE Component 2", fontsize=fontsize2
    )  # 使用较大的字体大小设置 Y 轴标签
    reward_title = "Explorative MSJRL" if reward == "rl_" else "Naive MSJRL"
    # 在图形底部中央添加模型名称和层信息，设置字体大小和斜体
    plt.figtext(
        0.5,
        0.05,
        f"{model_name.title()} (Layer {layer})\nNumber of shots: {shot}",
        ha="center",
        fontsize=fontsize1,
        style="italic",
    )

    # 移除 x 和 y 轴的刻度标签，使得图形更加清晰简洁
    plt.xticks([])
    plt.yticks([])

    # 调整图形布局，增加边距（pad=5.0）以避免标签或注释重叠
    plt.tight_layout(pad=5.0)

    # 如果 save 参数为 True，则保存图像到指定路径
    if save:
        # 创建保存路径，确保路径格式兼容文件系统，将不合法字符替换为下划线
        save_fig_path = f"./tsne_visualization_{reward}{dataset_name}_{shot}shot.pdf"

        # 保存图像，使用 'bbox_inches=tight' 确保图像边框紧贴内容，避免多余的空白
        plt.savefig(save_fig_path, bbox_inches="tight")

    # 返回绘制好的图像对象，便于在外部显示或进一步操作
    return plt


# def visualize_act_with_t_sne(model_name, tensors_flat, labels, dataset_name, num_samples, layer, perplexity =30, scores=None, save = False):
#   # Apply t-SNE
#   print("t-SNE visualization of the weights of the residual stream of the model")
#   tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
#   print("Fitting t-SNE...", [tensor for tensor in tensors_flat])

#   tensors_tsne = tsne.fit_transform(tensors_flat)

#   # Check the shape of tensors_tsne to ensure it matches expected dimensions
#   print("Shape of tensors_tsne:", tensors_tsne.shape)

#   # Ensure that the number of labels matches the number of samples in tensors_tsne
#   assert len(labels) == tensors_tsne.shape[0], "The number of labels must match the number of samples in tensors_tsne."

#   # Plot the t-SNE results with different colors for different labels
#   plt.figure(figsize=(10, 8))

#   # Create a color map
#   # colors = plt.colormaps.get_cmap('Accent')  # Get the colormap without specifying number of colors
#   # Map colors to unique occurrences in labels
#   unique_labels = np.unique(labels)
#   # colors = cm.get_cmap('viridis', len(unique_labels))  # Choose a colormap
#   # colors = cm.get_cmap('rainbow', len(unique_labels))  # Choose a colormap
#   colors = cm.get_cmap('tab20', len(unique_labels))  # Choose the 'tab20' colormap
#   color_map = {label: colors(i) for i, label in enumerate(unique_labels)}


#   # Plot each label category with a different color
#   for label in np.unique(labels):
#       indices = np.where(labels == label)[0]

#       if scores is not None:
#         for idx in indices:
#               base_color = np.array(color_map[label][:3])  # Extract RGB components
#               score_adjusted_color = base_color * 0.5 + base_color * 0.5 * scores[idx]  # Adjust brightness
#               alpha = 0.3 + 0.7 * scores[idx]  # Ensure minimum visibility for low scores
#               plt.scatter(tensors_tsne[idx, 0], tensors_tsne[idx, 1], label=label if idx == indices[0] else "",
#                           color=score_adjusted_color, alpha=alpha, s=100)
#       else:
#               plt.scatter(tensors_tsne[indices, 0], tensors_tsne[indices, 1], label=label,
#                     color=color_map[label], alpha=0.5, s=100)  # Mapping scores to alpha


#   plt.title(f"t-SNE Visualization of Residual Stream Weights, model={model_name}, layer={layer}, num_samples={num_samples}, perplexity={perplexity}")
#   plt.xlabel("t-SNE Component 1")
#   plt.ylabel("t-SNE Component 2")
#   plt.figtext(0.5, 0.01, f"Dataset: {dataset_name}", ha='center', fontsize=10, style='italic')  # Update dataset name here
#   plt.legend()

#   if save:
#     now = datetime.datetime.now().strftime("%d%m%Y%H:%M:%S")
#     save_fig_path = f'tsne_visualization_{dataset_name}_{now}.pdf'.replace('/', '_')  # Replace with your desired path
#     plt.savefig(save_fig_path)

#   plt.show()


# def visualize_act_with_t_sne(model_name, tensors_flat, labels, dataset_name, num_samples, layer, perplexity=30, scores=None, save=False):
#     print("t-SNE visualization of the weights of the residual stream of the model")
#     tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
#     tensors_tsne = tsne.fit_transform(tensors_flat)

#     print("Shape of tensors_tsne:", tensors_tsne.shape)
#     assert len(labels) == tensors_tsne.shape[0], "The number of labels must match the number of samples in tensors_tsne."

#     plt.figure(figsize=(10, 8))

#     # Use a distinct color map
#     unique_labels = np.unique(labels)
#     colors = plt.get_cmap('tab20b').colors
#     if len(unique_labels) > 20:
#         additional_colors = plt.get_cmap('tab20c').colors
#         colors = colors + additional_colors
#     color_map = {label: colors[i % len(colors)] for i, label in enumerate(unique_labels)}

#     for label in unique_labels:
#         indices = np.where(labels == label)[0]
#         plt.scatter(tensors_tsne[indices, 0], tensors_tsne[indices, 1], label=label,
#                     color=color_map[label], alpha=0.7, s=100)

#     plt.title(f"t-SNE Visualization of Residual Stream Weights, model={model_name}, layer={layer}, num_samples={num_samples}, perplexity={perplexity}")
#     plt.xlabel("t-SNE Component 1")
#     plt.ylabel("t-SNE Component 2")
#     plt.figtext(0.5, 0.01, f"Dataset: {dataset_name}", ha='center', fontsize=10, style='italic')
#     plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))

#     if save:
#         now = datetime.datetime.now().strftime("%d%m%Y%H:%M:%S")
#         save_fig_path = f'tsne_visualization_{dataset_name}_{now}.pdf'.replace('/', '_')
#         plt.savefig(save_fig_path)

#     plt.show()


def visualize_act_with_t_sne_interactive(
    model_name,
    tensors_flat,
    labels,
    dataset_name,
    num_samples,
    layer,
    perplexity=30,
    scores=None,
    save=False,
):
    """
    使用 t-SNE 和 Plotly 进行交互式可视化的函数。

    参数:
    - model_name: 模型名称，用于图形标题和保存路径。
    - tensors_flat: 输入张量，通常是模型层的输出，用于 t-SNE 降维。
    - labels: 每个样本的标签，用于分类和颜色区分。
    - dataset_name: 数据集名称，用于保存路径。
    - num_samples: 样本数量，显示在标题中。
    - layer: 当前可视化的模型层。
    - perplexity: t-SNE 参数，影响聚类的平滑性，默认为 30。
    - scores: 可选参数，可能用于根据某些得分调整点的大小（当前未使用）。
    - save: 是否将图像保存为 HTML 文件，默认为 False。
    """

    # 使用 t-SNE 对输入的张量进行降维，降到 2 个组件，用于可视化
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    tensors_tsne = tsne.fit_transform(tensors_flat)

    # 创建一个 Pandas DataFrame 来存储 t-SNE 降维后的结果和对应的标签
    df = pd.DataFrame(tensors_tsne, columns=["Component 1", "Component 2"])

    # 打印标签的长度和内容，调试用
    print("len", len(labels))
    print(labels)

    # 将标签添加到 DataFrame 中
    df["Label"] = labels

    # 如果有 scores 参数，则将其作为点的大小列添加到 DataFrame 中
    # 当前这部分代码被注释掉了，但可以用于未来扩展
    # if scores is not None:
    #     df['Score'] = scores
    #     size_column = 'Score'  # 使用 scores 列作为点的大小
    # else:
    #     size_column = None  # 如果没有 scores，则不调整点的大小

    # 使用 Plotly 创建一个交互式散点图
    fig = px.scatter(
        df,  # 数据来源
        x="Component 1",  # t-SNE 第一个组件作为 x 轴
        y="Component 2",  # t-SNE 第二个组件作为 y 轴
        color="Label",  # 使用标签来区分颜色
        title=f"t-SNE Visualization of Residual Stream Weights<br>(Model: {model_name}, Layer: {layer}, N = {num_samples}, Perplexity = {perplexity})",  # 标题
        hover_data=["Label"],  # 当鼠标悬停时显示的额外信息
        # size=size_column  # 点的大小（当前未使用）
    )

    # 更新图形布局，设置 x 轴和 y 轴的标题，以及图例标题
    fig.update_layout(
        xaxis_title="t-SNE Component 1",  # X轴标签
        yaxis_title="t-SNE Component 2",  # Y轴标签
        legend_title="Label",  # 图例标题
        width=900,  # 图像宽度
        height=700,  # 图像高度
    )

    # 显示交互式图形
    # fig.show()

    # 如果 save 参数为 True，将图像保存为 HTML 文件
    if save:
        # 获取当前时间，格式为 "日月年时分秒"，用于文件名
        now = datetime.datetime.now().strftime("%d%m%Y%H:%M:%S")

        # 创建保存路径，确保路径格式正确，将不合法字符替换为下划线
        save_fig_path = f"tsne_visualization_{dataset_name}.html".replace("/", "_")

        # 将生成的交互式图表保存为 HTML 文件
        fig.write_html(save_fig_path)


def visualize_act_with_pca(
    model_name,
    tensors_flat,
    labels,
    dataset_name,
    num_samples,
    layer,
    shot,
    reward,
    scores=None,
    save=False,
):
    """
    使用 PCA 进行模型权重的可视化，并绘制 2D 图形。

    参数:
    - model_name: 模型名称，用于图形标题和保存路径。
    - tensors_flat: 输入张量，通常是模型层的输出。
    - labels: 每个张量对应的标签，用于区分类别。
    - dataset_name: 数据集名称，用于保存路径和图形注释。
    - num_samples: 样本数量，用于图形标题。
    - layer: 当前可视化的模型层。
    - scores: 可选的得分列表，用于调整点的透明度和亮度。
    - save: 是否保存生成的图像，默认为 False。
    """

    # 使用 PCA 将输入张量降维到 2 个组件，用于 2D 可视化
    pca = PCA(n_components=2)
    tensors_pca = pca.fit_transform(tensors_flat)

    # 确保标签数量与 PCA 结果的样本数量相同，如果不匹配则抛出断言错误
    assert (
        len(labels) == tensors_pca.shape[0]
    ), "The number of labels must match the number of samples in tensors_pca."

    # 创建一个新的图形窗口，大小为 10x8 英寸
    plt.figure(figsize=(10, 8))

    # 获取颜色映射
    # 使用 'viridis' colormap, 并根据标签数量生成相应的颜色
    unique_labels = np.unique(labels)
    colors = cm.get_cmap("viridis", len(unique_labels))
    color_map = {label: colors(i) for i, label in enumerate(unique_labels)}

    # 遍历每个唯一的标签，并绘制相应类别的点
    for label in np.unique(labels):
        # 获取该标签的样本索引
        indices = np.where(labels == label)[0]

        # 如果提供了 scores，则根据 scores 调整颜色亮度和透明度
        if scores is not None:
            for idx in indices:
                # 获取基础颜色（RGB 值）
                base_color = np.array(color_map[label][:3])

                # 根据 score 调整颜色亮度
                score_adjusted_color = base_color * 0.5 + base_color * 0.5 * scores[idx]

                # 根据 score 调整透明度
                alpha = (
                    0.3 + 0.7 * scores[idx]
                )  # 确保低分样本仍然可见，alpha 最低为 0.3

                # 绘制单个样本点
                plt.scatter(
                    tensors_pca[idx, 0],
                    tensors_pca[idx, 1],
                    label=(
                        label if idx == indices[0] else ""
                    ),  # 确保每个标签只在图例中显示一次
                    color=score_adjusted_color,
                    alpha=alpha,  # 设置透明度
                    s=100,  # 点的大小
                )
        else:
            # 如果没有 scores，直接绘制类别点，颜色固定，透明度为 0.5
            plt.scatter(
                tensors_pca[indices, 0],
                tensors_pca[indices, 1],
                label=label,  # 标签类别
                color=color_map[label],  # 颜色
                alpha=0.5,  # 透明度
                s=100,  # 点的大小
            )

    # 设置图像的标题，包含模型名称、层数和样本数量
    plt.title(
        f"PCA Visualization of Residual Stream Weights, model={model_name}, layer={layer}, num_samples={num_samples}"
    )

    # 设置 x 轴和 y 轴的标签
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")

    # 在图形底部中央添加数据集名称信息，使用斜体
    plt.figtext(
        0.5, 0.01, f"Dataset: {dataset_name}", ha="center", fontsize=10, style="italic"
    )

    # 添加图例，显示标签颜色对应关系
    plt.legend()

    # 如果 save 参数为 True，将图像保存为 PDF 文件
    if save:
        # 获取当前时间，格式为 "日月年时分秒"，用于文件名
        now = datetime.datetime.now().strftime("%d%m%Y%H:%M:%S")  # 获取当前时间字符串
        # 构建保存路径，确保文件名合法（替换不合法字符）
        save_fig_path = (
            f"pca_visualization_{reward}{dataset_name}_{shot}shot.pdf".replace("/", "_")
        )

        # 保存图像为 PDF 文件，使用 'bbox_inches='tight'' 确保图像边框紧贴内容，避免多余的空白
        plt.savefig(save_fig_path, bbox_inches="tight")

    # 显示绘制好的图像
    # plt.show()


def visualize_act_with_pca_interactive(
    model_name,
    tensors_flat,
    labels,
    dataset_name,
    num_samples,
    layer,
    cluster_label=None,
    scale_factor=0.1,
    scores=None,
    save=False,
):
    """
    使用PCA对张量数据进行可视化，并创建交互式散点图。

    参数:
    model_name: 模型名称
    tensors_flat: 扁平化的张量数据
    labels: 数据标签
    dataset_name: 数据集名称
    num_samples: 样本数量
    layer: 层数
    cluster_label: 特定的聚类标签（可选）
    scale_factor: 缩放因子（可选，默认为0.1）
    scores: 分数数据（可选）
    save: 是否保存图表（可选，默认为False）
    """

    # 应用PCA降维到2个主成分
    pca = PCA(n_components=2)
    tensors_pca = pca.fit_transform(tensors_flat)

    # 创建包含PCA结果和标签的DataFrame
    df = pd.DataFrame(tensors_pca, columns=["PCA Component 1", "PCA Component 2"])
    df["Label"] = labels

    # 如果提供了特定的聚类标签，调整这些点使它们更靠近
    if cluster_label is not None:
        cluster_indices = df["Label"] == cluster_label
        # 缩放坐标以使点更靠近
        df.loc[cluster_indices, ["PCA Component 1", "PCA Component 2"]] *= scale_factor

    # 如果提供了分数，将其包含在DataFrame中并增加点的大小
    if scores is not None:
        df["Score"] = scores
        size_column = "Score"
    else:
        size_column = None

    # 使用Plotly创建交互式散点图
    fig = px.scatter(
        df,
        x="PCA Component 1",
        y="PCA Component 2",
        color="Label",
        title=f"PCA Visualization of Residual Stream Weights<br>(Model: {model_name}, Layer: {layer}, N = {num_samples})",
        hover_data=["Label"],
        size=size_column,
        size_max=20,  # 增加点的最大尺寸
    )

    # 更新图表布局
    fig.update_layout(
        xaxis_title="PCA Component 1",
        yaxis_title="PCA Component 2",
        legend_title="Label",
        width=800,
        height=600,
    )

    # 显示图表
    # fig.show()

    # 如果save为True，将图表保存为HTML文件
    if save:
        now = datetime.datetime.now().strftime("%d%m%Y%H:%M:%S")
        save_fig_path = f"pca_visualization_{dataset_name}.html".replace("/", "_")
        fig.write_html(save_fig_path)
