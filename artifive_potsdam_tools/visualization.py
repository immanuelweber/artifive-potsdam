import matplotlib.patches as patches
import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
import torch as th
import torchvision as tv


def is_list_like(obj):
    if isinstance(obj, th.Tensor):
        return obj.dim() > 0
    elif isinstance(obj, str):
        return 0
    is_iter_colection = hasattr(obj, "__iter__") or hasattr(obj, "__getitem__")
    has_len = hasattr(obj, "__len__")
    return is_iter_colection and has_len


def listify(obj):
    if is_list_like(obj):
        return obj
    if obj is None:
        return []
    return [obj]


def draw_outline(plot_object, lw, color="black"):
    plot_object.set_path_effects(
        [patheffects.Stroke(linewidth=lw, foreground=color), patheffects.Normal()]
    )


def draw_rectangle(
    axis,
    rectangle,
    color="white",
    lw=2,
    outline_color="black",
    outline_lw=4,
    fill=False,
    fillcolor=None,
    linestyle="solid",
):
    xy = rectangle[:2]
    wh = rectangle[2:] - rectangle[:2]
    patch = axis.add_patch(
        patches.Rectangle(
            xy,
            *wh,
            fill=fill,
            facecolor=fillcolor,
            edgecolor=color,
            lw=lw,
            linestyle=linestyle,
        )
    )
    draw_outline(patch, outline_lw, outline_color)
    return axis


def draw_rectangles(
    axis,
    rectangles,
    color="white",
    lw=2,
    outline_color="black",
    outline_lw=4,
    fill=False,
    fillcolor=None,
    linestyle="solid",
):
    for r in rectangles:
        draw_rectangle(
            axis, r, color, lw, outline_color, outline_lw, fill, fillcolor, linestyle
        )
    return axis


def draw_text(
    axis, xy, text, size=14, color="white", outline_color="black", outline_lw=1
):
    text_object = axis.text(
        *xy, text, verticalalignment="top", color=color, fontsize=size, weight="bold"
    )
    draw_outline(text_object, outline_lw, outline_color)
    text_object.set_clip_on(True)


def __get_translations(h, w, n_samples, n_cols, padding):
    txs, tys = [], []
    j = -1
    for i in range(n_samples):
        txs.append((w + padding) * (i % n_cols))
        if i % n_cols == 0:
            j += 1
        tys.append((h + padding) * j)
    return txs, tys


def draw_texts(
    axis, texts, positions, size=14, color="white", outline_lw=4, outline_color="black"
):
    texts = listify(texts)
    for t, p in zip(texts, positions):
        draw_text(axis, p, t, size, color, outline_color, outline_lw)
    return axis


def draw_batch(
    rasters,
    boxes=None,
    labels=None,
    norm_mean=None,
    norm_std=None,
    label_map=None,
    n_cols=8,
    padding=2,
    figsize=(16, 8),
    axis=None,
):
    if axis is None:
        _, axis = plt.subplots(figsize=figsize)

    gridded_images = tv.utils.make_grid(rasters, nrow=n_cols, padding=padding)
    gridded_images = gridded_images.permute(1, 2, 0)
    if norm_mean is not None and norm_std is not None:
        gridded_images = gridded_images.mul(th.as_tensor(norm_std)).add(
            th.as_tensor(norm_mean)
        )
    axis.imshow(gridded_images.numpy())
    axis.axis(False)

    if label_map:
        inverse_label_map = {v: k for k, v in label_map.items()}

    _, h, w = rasters[0].shape
    txs, tys = __get_translations(h, w, len(rasters), n_cols, padding)
    if boxes is not None:
        for id, (bxs, tx, ty) in enumerate(zip(boxes, txs, tys)):
            bxs = bxs.cpu() + th.as_tensor([tx, ty, tx, ty])
            draw_rectangles(axis, bxs)
            if labels is not None:
                lbls = labels[id].tolist()
                if label_map:
                    lbls = [inverse_label_map[lbl] for lbl in lbls]
                draw_texts(axis, lbls, bxs[:, :2])
