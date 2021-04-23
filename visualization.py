import matplotlib.patches as patches
import matplotlib.patheffects as patheffects


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