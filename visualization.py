import matplotlib.patches as patches
import matplotlib.patheffects as patheffects
import torch


def is_list_like(obj):
    if isinstance(obj, torch.Tensor):
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


def draw_texts(
    axis, texts, positions, size=14, color="white", outline_lw=4, outline_color="black"
):
    texts = listify(texts)
    for t, p in zip(texts, positions):
        draw_text(axis, p, t, size, color, outline_color, outline_lw)
    return axis
