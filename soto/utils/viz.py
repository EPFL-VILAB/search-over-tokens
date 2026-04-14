"""Visualisation helpers for SoTo notebooks and scripts."""

import textwrap

import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

__all__ = ["load_img", "show_images", "show_rows", "show_progressive", "show_side_by_side_progressive", "show_search_tree"]


def load_img(path, img_size=256, device="cpu"):
    """Load an image and preprocess it for FlexTok.

    Resizes the shorter side to img_size, center-crops to a square, converts
    to a float tensor normalized to [-1, 1], and adds a batch dimension.

    Args:
        path: Path to the image file.
        img_size: Output spatial resolution (default 256).
        device: Target device string (default "cpu").

    Returns:
        Tensor of shape [1, 3, img_size, img_size] normalized to [-1, 1].
    """
    from PIL import Image as _Image
    img = _Image.open(path).convert("RGB")
    img = TF.center_crop(TF.resize(img, img_size), img_size)
    tensor = TF.normalize(TF.to_tensor(img), [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    return tensor.unsqueeze(0).to(device)


def show_images(
    images,
    titles=None,
    overlay_labels=None,
    overlay_prefix=None,
    prompt=None,
    ncols=None,
    first_row_cols=None,
    main_title=None,
    save_path=None,
):
    """Display a flat grid of images with optional titles, score/count overlays,
    and a prompt text box.

    This is the primary single-image display function.  All other helpers are
    thin wrappers around ``show_images`` or ``show_rows``.

    Args:
        images: List of PIL images.
        titles: Optional list of bold titles shown *above* each image (e.g.
            method names like ``["Direct AR", "Beam Search"]``).
        overlay_labels: Optional list of values (``float``, ``int``, ``str``, or
            ``None``) drawn as a semi-transparent white badge in the **top-right
            corner** of each image.  Floats are formatted to 2 decimal places;
            ints and strings are used as-is.  ``None`` entries are skipped (no
            overlay).
        overlay_prefix: Optional prefix prepended to each overlay label, e.g.
            ``"ImageReward"`` → ``"ImageReward: 0.543"``.  Pass ``None`` to
            show the bare value (useful for token counts).
        prompt: If given, a white text-box panel with the wrapped prompt is
            prepended as the first panel in the grid.
        ncols: Number of columns.  Defaults to ``min(total_panels, 6)``.
        first_row_cols: If set (e.g. 2), the first row has this many columns
            (plus prompt if given); remaining panels use ``ncols`` for layout.
        main_title: Optional figure-level ``suptitle``.
        save_path: If given, save the figure to this path at 150 dpi.
            Defaults to ``None`` (no save).
    """
    n = len(images)
    has_prompt = prompt is not None
    total = n + (1 if has_prompt else 0)

    if first_row_cols is not None:
        n_in_first = (1 if has_prompt else 0) + first_row_cols
        n_remaining = total - n_in_first
        ncols_rest = ncols or min(max(n_remaining, 1), 6)
        ncols_grid = max(first_row_cols, ncols_rest)
        nrows_rest = (n_remaining + ncols_rest - 1) // ncols_rest if n_remaining > 0 else 0
        nrows = 1 + nrows_rest
        ncols = ncols_grid
    else:
        ncols = ncols or min(total, 6)
        nrows = (total + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.8 * nrows))
    axes = np.array(axes).flatten()

    # ── Map panel index to axis index when first_row_cols is used ──────────────
    def panel_to_ax_idx(panel_idx):
        if first_row_cols is None:
            return panel_idx
        if panel_idx < n_in_first:
            return panel_idx  # row 0
        local = panel_idx - n_in_first
        r = 1 + local // ncols_rest
        c = local % ncols_rest
        return r * ncols + c

    # ── Optional prompt box (first panel) ────────────────────────────────────
    offset = 0
    if has_prompt:
        ax = axes[panel_to_ax_idx(0)]
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_facecolor("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("lightgray")
            spine.set_linewidth(1)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        ax.text(
            0.5, 0.5, textwrap.fill(prompt, width=22),
            ha="center", va="center", fontsize=12, linespacing=1.4,
            transform=ax.transAxes,
        )
        ax.set_title("Prompt", fontsize=12)
        offset = 1

    # ── Image panels ─────────────────────────────────────────────────────────
    for i, img in enumerate(images):
        ax = axes[panel_to_ax_idx(i + offset)]
        ax.imshow(img)
        ax.axis("off")

        if titles:
            ax.set_title(titles[i], fontsize=12, fontweight="bold", pad=6)

        if overlay_labels is not None:
            val = overlay_labels[i]
            if val is None:
                pass  # no overlay for this image
            else:
                if isinstance(val, str):
                    val_str = val
                elif isinstance(val, (int, np.integer)):
                    val_str = str(int(val))
                else:
                    val_str = f"{float(val):.2f}"   # handles float, torch.Tensor, etc.
                text = f"{overlay_prefix}: {val_str}" if overlay_prefix else val_str
                ax.text(
                0.97, 0.97, text,
                transform=ax.transAxes, ha="right", va="top",
                fontsize=11, fontweight="bold", color="black",
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="none",
                          boxstyle="round,pad=0.3"),
            )

    # Hide unused axes
    if first_row_cols is not None:
        used = {panel_to_ax_idx(i) for i in range(total)}
        for i in range(len(axes)):
            if i not in used:
                axes[i].set_visible(False)
    else:
        for i in range(total, len(axes)):
            axes[i].set_visible(False)

    if main_title:
        fig.suptitle(main_title, fontsize=12, fontweight="bold", y=1.01)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()


def show_rows(row_data, title=None, col_titles=None, save_path=None):
    """Display images arranged in labeled rows.

    A 2-D grid where every row is one named group (e.g. a search method or a
    tokenizer type) and every column is a step or token-count checkpoint.
    Each cell carries a small white overlay label in its top-right corner.

    This function underlies both the intermediate search-steps comparison and
    the FlexTok/GridTok side-by-side progressive decoding plot.

    Args:
        row_data: List of ``(row_label, images, overlay_labels)`` tuples:

            * ``row_label`` – string shown as the bold y-axis label of the row.
            * ``images`` – list of PIL images for this row.
            * ``overlay_labels`` – list of strings, one per image, drawn as a
              semi-transparent white badge in the top-right corner.
              Pass ``None`` or a list of empty strings to suppress badges.

        title: Optional figure ``suptitle``.
        col_titles: Optional list of strings shown as column headers above the
            first row (one per column). Use instead of overlay_labels when the
            label belongs to the column, not the individual image.
        save_path: If given, save the figure to this path at 150 dpi.
            Defaults to ``None`` (no save).

    Example::

        show_rows(
            [
                ("Direct AR",   ar_images,   [f"{t} tok" for t in token_counts]),
                ("Beam Search", beam_images, [f"{s:.2f}" for s in beam_scores]),
                ("Lookahead",   la_images,   [f"{s:.2f}" for s in la_scores]),
            ],
            title="Intermediate steps",
        )
    """
    n_cols = max(len(imgs) for _, imgs, _ in row_data)
    n_rows = len(row_data)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 3.6 * n_rows))
    axes = np.array(axes).reshape(n_rows, n_cols)

    for row_idx, (row_label, images, overlay_labels) in enumerate(row_data):
        for col_idx in range(n_cols):
            ax = axes[row_idx][col_idx]
            if col_idx < len(images) and images[col_idx] is not None:
                ax.imshow(images[col_idx])
                label = overlay_labels[col_idx] if overlay_labels else ""
                if label:
                    ax.text(
                        0.97, 0.97, label,
                        transform=ax.transAxes, ha="right", va="top",
                        fontsize=11, fontweight="bold", color="black",
                        bbox=dict(facecolor="white", alpha=0.75, edgecolor="none",
                                  boxstyle="round,pad=0.3"),
                    )
            else:
                ax.set_visible(False)
                continue
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            for spine in ax.spines.values():
                spine.set_visible(False)
            if col_idx == 0:
                ax.set_ylabel(row_label, fontsize=11, fontweight="bold", labelpad=8)

    if col_titles:
        for col_idx, col_title in enumerate(col_titles[:n_cols]):
            axes[0, col_idx].set_title(col_title, fontsize=11, fontweight="bold", pad=6)

    if title:
        fig.suptitle(title, fontsize=11, y=1.01)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()


def show_search_tree(
    all_results,
    prompt: str = "",
    n_show: int = 5,
    img: float = 1.25,
    gap_x: float = 0.12,
    gap_y: float = 0.55,
    start_gap: float = 0.3,
    save_path: str = None,
):
    """Visualise a multi-step beam search as a top-to-bottom tree.

    Each row is one search step; columns are the top-scoring candidates.
    Children of the same surviving beam are grouped together and placed
    roughly below their parent.  Downward arrows connect survivors to their
    children.  A "Start" box sits above the first row with arrows to all
    step-0 candidates.

    Args:
        all_results: List of ``SearchResult`` objects, one per step.
        prompt:      Text prompt shown at the top of the figure.
        n_show:      Number of candidates shown per row (must equal
                     ``len(result.images)`` for every result).
        img:         Image cell size in inches.
        gap_x:       Horizontal gap between images within a row (inches).
        gap_y:       Vertical gap between rows — space for arrows (inches).
        start_gap:   Gap between the Start box and the first row (inches).
        save_path:   If given, save to this path at 150 dpi.
    """
    import torch
    from matplotlib.patches import FancyArrowPatch
    import matplotlib.patches as mpatches

    n_steps = len(all_results)
    START_H = 0.55
    ML, MR  = 0.9, 0.2
    MT, MB  = 0.35, 0.15

    row_w = n_show * img + (n_show - 1) * gap_x
    fig_w = ML + row_w + MR
    fig_h = MB + n_steps * img + (n_steps - 1) * gap_y + start_gap + START_H + MT

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")

    def fx(x): return x / fig_w
    def fy(y): return y / fig_h

    def col_x(slot):   return ML + slot * (img + gap_x)
    def step_y(step):  return MB + (n_steps - 1 - step) * (img + gap_y)

    def is_survivor(result, col):
        import torch as _torch
        return any(_torch.equal(result.display_tokens[col], result.tokens[b])
                   for b in range(result.tokens.size(0)))

    # ── Column slot assignment: group children under their parent ─────────────
    step_slots = {0: {c: c for c in range(n_show)}}
    for s in range(1, n_steps):
        prev, curr = all_results[s - 1], all_results[s]
        surv_cols  = sorted([c for c in range(n_show) if is_survivor(prev, c)])
        prev_slots = step_slots[s - 1]
        child_of   = {}
        for col_dst in range(n_show):
            prefix = curr.display_tokens[col_dst][:-1]
            for col_src in surv_cols:
                if torch.equal(prev.display_tokens[col_src], prefix):
                    child_of[col_dst] = col_src
                    break
        sorted_cols = sorted(range(n_show),
                             key=lambda c: (prev_slots.get(child_of.get(c, -1), 999), c))
        step_slots[s] = {dc: slot for slot, dc in enumerate(sorted_cols)}

    # ── Shaded row backgrounds ────────────────────────────────────────────────
    band_colors = ["#dde8f5", "#cfdded", "#c1d3e5", "#b3c8dd", "#a5bed5"]
    for s in range(n_steps):
        pad  = 0.08
        rect = mpatches.FancyBboxPatch(
            (fx(ML - pad), fy(step_y(s) - pad)),
            fx(row_w + 2*pad), fy(img + 2*pad),
            boxstyle="round,pad=0.01", transform=fig.transFigure,
            facecolor=band_colors[s % len(band_colors)], edgecolor="none", zorder=0,
        )
        fig.add_artist(rect)

    # ── Images + ✓/✗ overlays + score ────────────────────────────────────────
    for s, result in enumerate(all_results):
        for col in range(n_show):
            slot  = step_slots[s][col]
            x, y  = col_x(slot), step_y(s)
            ax    = fig.add_axes([fx(x), fy(y), fx(img), fy(img)])
            ax.imshow(result.images[col])
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0)

            ax.text(0.5, 0.03, f"{result.scores[col].item():.2f}",
                    transform=ax.transAxes, fontsize=6, color="white",
                    ha="center", va="bottom", zorder=5,
                    bbox=dict(facecolor="black", alpha=0.4, edgecolor="none", pad=1))

    # ── Arrows: survivors → children ─────────────────────────────────────────
    for s in range(n_steps - 1):
        for col_src in range(n_show):
            if not is_survivor(all_results[s], col_src):
                continue
            tok_src = all_results[s].display_tokens[col_src]
            for col_dst in range(n_show):
                prefix = all_results[s + 1].display_tokens[col_dst][:-1]
                if torch.equal(prefix, tok_src):
                    x0 = col_x(step_slots[s][col_src])     + img / 2
                    y0 = step_y(s)
                    x1 = col_x(step_slots[s + 1][col_dst]) + img / 2
                    y1 = step_y(s + 1) + img
                    fig.add_artist(FancyArrowPatch(
                        (fx(x0), fy(y0)), (fx(x1), fy(y1)),
                        transform=fig.transFigure, arrowstyle="-|>", color="#444",
                        lw=1.3, mutation_scale=10, connectionstyle="arc3,rad=0.0", zorder=4,
                    ))

    # ── Start box ─────────────────────────────────────────────────────────────
    start_bottom = step_y(0) + img + start_gap
    mid_x        = ML + row_w / 2
    box_w, box_h = 1.2, START_H * 0.85
    ax_s = fig.add_axes([fx(mid_x - box_w / 2), fy(start_bottom), fx(box_w), fy(box_h)])
    ax_s.set_xticks([]); ax_s.set_yticks([])
    ax_s.text(0.5, 0.5, "Start", ha="center", va="center",
              fontsize=10, fontweight="bold", transform=ax_s.transAxes)
    for spine in ax_s.spines.values():
        spine.set_linewidth(2); spine.set_edgecolor("#333")

    for col in range(n_show):
        slot = step_slots[0][col]
        fig.add_artist(FancyArrowPatch(
            (fx(mid_x), fy(start_bottom)),
            (fx(col_x(slot) + img / 2), fy(step_y(0) + img)),
            transform=fig.transFigure, arrowstyle="-|>", color="#444",
            lw=1.3, mutation_scale=10, connectionstyle="arc3,rad=0.0", zorder=4,
        ))

    # ── Row labels ────────────────────────────────────────────────────────────
    suffixes = ["st", "nd", "rd"] + ["th"] * (n_steps - 3)
    for s in range(n_steps):
        fig.text(fx(ML - 0.12), fy(step_y(s) + img / 2),
                 f"{s + 1}{suffixes[s]} token",
                 ha="right", va="center", fontsize=8.5, fontweight="bold", color="#222")

    # ── Prompt ────────────────────────────────────────────────────────────────
    if prompt:
        fig.text(0.5, fy(start_bottom + START_H + 0.05),
                 f"Prompt: {prompt}", ha="center", va="bottom",
                 fontsize=7.5, fontstyle="italic", color="#333")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.show()


def show_progressive(images, token_counts, prompt=None):
    """Show images decoded from increasing token-count prefixes.

    Token counts are shown as overlaid labels on each image.
    Wraps :func:`show_images`.

    Args:
        images: List of PIL images, one per token count.
        token_counts: Matching list of token counts used to decode each image.
        prompt: Optional text prompt shown as the first panel.
    """
    labels = [f"{t} Token{'s' if t != 1 else ''}" for t in token_counts]
    show_images(images, overlay_labels=labels, prompt=prompt)


def show_side_by_side_progressive(flex_imgs, grid_imgs, flex_labels, grid_labels, prompt):
    """Compare FlexTok and GridTok progressive decoding side by side.

    Wraps :func:`show_rows` with two rows:

    * Row 0 — FlexTok (1D coarse-to-fine): global semantics visible from token 1.
    * Row 1 — GridTok (2D raster scan): only the top-left corner visible early on.

    This directly illustrates the paper's key insight: ordered 1D tokens carry
    semantic meaning in partial prefixes, making verifier-guided search effective.

    Args:
        flex_imgs: List of PIL images from FlexTok progressive decoding.
        grid_imgs: List of PIL images from GridTok progressive decoding (same length).
        flex_labels: Overlay labels for FlexTok columns (e.g. ``["1 tok", "4 tok", ...]``).
        grid_labels: Overlay labels for GridTok columns.
        prompt: Text prompt used to generate the images; shown in the figure title.
    """
    assert len(flex_imgs) == len(grid_imgs), "Both image lists must have the same length"
    short = textwrap.fill(prompt, width=80)
    show_rows(
        [
            ("FlexTok\n(1D coarse-to-fine)", flex_imgs, flex_labels),
            ("GridTok\n(2D raster scan)",    grid_imgs, grid_labels),
        ],
        title=f'Progressive Decoding Comparison\n"{short}"',
    )


