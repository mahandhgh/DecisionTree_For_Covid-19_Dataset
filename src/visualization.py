import matplotlib.pyplot as plt


def plot_custom_tree(node, depth=0, x=0.5, y=1.0, dx=0.1, dy=0.1, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    if node.leaf:
        ax.text(
            x,
            y,
            f"Class: {node.predicted_class}\nSamples: {node.num_samples}",
            ha="center",
            va="center",
            bbox=dict(
                facecolor="lightgreen",
                edgecolor="black",
            ),
            fontsize=10,
        )
    else:
        ax.text(
            x,
            y,
            f"Feature {node.feature_index}\n"
            f"Thresh: {node.threshold:.2f}\n"
            f"Samples: {node.num_samples}",
            ha="center",
            va="center",
            bbox=dict(facecolor="lightblue", edgecolor="black"),
            fontsize=10,
        )

        if node.left:
            ax.plot(
                [x, x - dx],
                [y, y - dy],
                color="black",
            )
            plot_custom_tree(node.left, depth + 1, x - dx, y - dy, dx / 2, dy, ax)

        if node.right:
            ax.plot(
                [x, x + dx],
                [y, y - dy],
                color="black",
            )
            plot_custom_tree(node.right, depth + 1, x + dx, y - dy, dx / 2, dy, ax)

    return ax
