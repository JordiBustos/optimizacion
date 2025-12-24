from htbuilder import (
    HtmlElement,
    div,
    a,
    p,
    img,
    styles,
)
from htbuilder.units import percent, px
import streamlit as st


def image(src_as_string, **style):
    return img(src=src_as_string, style=styles(**style))


def link(link, text, **style):
    return a(_href=link, _target="_blank", style=styles(**style))(text)


def layout(*args):

    style = """
    <style>
      # MainMenu {visibility: hidden;}
      footer {visibility: hidden;}
     .stApp { bottom: 40px; }
    </style>
    """

    style_div = styles(
        position="fixed",
        left=0,
        bottom=0,
        margin=px(0, 0, 0, 0),
        width=percent(100),
        text_align="center",
        height="auto",
        opacity=1,
        background_color="white",
        font_size=px(12),
        border_top="1px solid #eaeaea",
        color="#555",
        padding=px(5, 5, 5, 5),
    )

    style_p = styles(
        padding=px(5, 5, 5, 5),
        margin=0
    )

    body = p(style=style_p)
    foot = div(style=style_div)(body)

    st.markdown(style, unsafe_allow_html=True)

    for arg in args:
        if isinstance(arg, str):
            body(arg)

        elif isinstance(arg, HtmlElement):
            body(arg)

    st.markdown(str(foot), unsafe_allow_html=True)


def footer():
    myargs = [
        "Entrega final - 2025 | ",
        link("https://github.com/jordiBustos", "@JordiBustos"),
    ]
    layout(*myargs)
