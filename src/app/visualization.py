import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


def show_final_results(results):
    """Visualize the final ranked images and their best matching detection."""

    n = len(results)

    if n == 0:
        print("No matching images found.")
        return

    fig, axes = plt.subplots(
        1,
        n,
        figsize=(5 * n, 5),
    )

    if n == 1:
        axes = [axes]

    for ax, result in zip(axes, results):
        image = Image.open(
            result["image_path"]
        ).convert("RGB")

        ax.imshow(image)

        detection = result["best_detection"]

        x, y, w, h = detection["bbox"]

        rectangle = patches.Rectangle(
            (x, y),
            w,
            h,
            linewidth=2,
            fill=False,
        )

        ax.add_patch(rectangle)

        ax.text(
            x,
            y,
            f'{detection["class_name"]} '
            f'{detection["confidence"]:.2f}',
        )

        ax.set_title(
            f'CLIP: {result["clip_score"]:.3f}\n'
            f'Final: {result["final_score"]:.3f}'
        )

        ax.axis("off")

    plt.show()
