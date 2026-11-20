"""Shared settings and helpers for the Plotly HTML previews written by primitives."""

import plotly.io as pio
from plotly.io._base_renderers import open_html_in_browser

# Sans-serif font for preview text.
FONT_FAMILY = '"Helvetica Neue", Helvetica, Arial, sans-serif'

# MathJax build for previews that render LaTeX; pass it as include_mathjax to write_html, and use
# show_figure instead of fig.show. Plotly's default ('cdn') is tex-svg, which loads some TeX
# extensions (e.g. \boldsymbol) on demand. When several texts using the same extension render at
# once, that on-demand load races and the macro stays undefined for the whole page (shown as red
# raw text). The -full build includes every TeX extension up front.
MATHJAX_URL = 'https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.2.2/es5/tex-svg-full.min.js'


def show_figure(fig, post_script=None):
    """
    Show a figure like fig.show, but with MATHJAX_URL when the default renderer is 'browser'.

    Plotly's browser renderer always writes include_mathjax='cdn' and silently ignores an
    include_mathjax argument to fig.show, so this builds the same page itself with MATHJAX_URL.
    Other renderers (e.g. notebooks) go through fig.show unchanged.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        The figure to show.
    post_script : str, optional
        JavaScript run after the plot is drawn, as in fig.show / write_html.
    """
    if pio.renderers.default == 'browser':
        html = fig.to_html(
            include_plotlyjs=True, include_mathjax=MATHJAX_URL, post_script=post_script, full_html=True,
            default_width='100%', default_height='100%',
        )
        open_html_in_browser(html)
    else:
        fig.show(post_script=post_script)
