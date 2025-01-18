# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022-2025)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np

import streamlit as st

LOREM_IPSUM = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua."

BLACK_IMG = np.repeat(0, 601350).reshape(633, 950)

# Basic columns:
c1, c2, c3 = st.columns(3)

c1.write(LOREM_IPSUM)
c2.write(LOREM_IPSUM)
c3.write(LOREM_IPSUM)

# Only fill in the last column and keep the others empty
c1, c2, c3 = st.columns(3)
c3.write(LOREM_IPSUM)

col1, col2 = st.columns(2, border=True)
with col1:
    st.metric("Temperature", "72°F", "2%")
with col2:
    st.metric("Pressure", "30.2 in", "-4%")
    st.slider("Slider", 0, 100, 50)

with st.expander("Variable-width columns (relative numbers)", expanded=True):
    for c in st.columns([0.6, 0.3, 0.1]):
        c.image(BLACK_IMG)

with st.expander("Variable-width columns (absolute numbers)", expanded=True):
    for c in st.columns((1, 2, 3, 4)):
        c.image(BLACK_IMG)

# Various column gaps
with st.expander("Column gap small", expanded=True):
    c4, c5, c6 = st.columns(3, gap="small")
    c4.image(BLACK_IMG)
    c5.image(BLACK_IMG)
    c6.image(BLACK_IMG)

with st.expander("Column gap medium", expanded=True):
    c7, c8, c9 = st.columns(3, gap="medium")
    c7.image(BLACK_IMG)
    c8.image(BLACK_IMG)
    c9.image(BLACK_IMG)

with st.expander("Column gap large", expanded=True):
    c10, c11, c12 = st.columns(3, gap="large")
    c10.image(BLACK_IMG)
    c11.image(BLACK_IMG)
    c12.image(BLACK_IMG)

with st.expander("Nested columns - one level", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        subcol1, subcol2 = st.columns(2)
        subcol1.write(LOREM_IPSUM)
        subcol2.write(LOREM_IPSUM)
        st.write("")
        st.write(LOREM_IPSUM)

    with col2:
        subcol1, subcol2 = st.columns(2)
        subcol1.write(LOREM_IPSUM)
        subcol2.write(LOREM_IPSUM)
        st.write("")
        subcol1, subcol2 = st.columns(2)
        subcol1.write(LOREM_IPSUM)
        subcol2.write(LOREM_IPSUM)

with st.expander("Vertical alignment - top", expanded=True):
    col1, col2, col3 = st.columns(3, vertical_alignment="top")
    col1.text_input("Text input (top)")
    col2.button("Button (top)", use_container_width=True)
    col3.checkbox("Checkbox (top)")

with st.expander("Vertical alignment - center", expanded=True):
    col1, col2, col3 = st.columns(3, vertical_alignment="center")
    col1.text_input("Text input (center)")
    col2.button("Button (center)", use_container_width=True)
    col3.checkbox("Checkbox (center)")

with st.expander("Vertical alignment - bottom", expanded=True):
    col1, col2, col3 = st.columns(3, vertical_alignment="bottom")
    col1.text_input("Text input (bottom)")
    col2.button("Button (bottom)", use_container_width=True)
    col3.checkbox("Checkbox (bottom)")

if st.button("Nested columns - two levels (raises exception)"):
    col1, col2 = st.columns(2)
    with col1:
        subcol1, subcol2 = st.columns(2)
        with subcol1:
            subcol1.write(LOREM_IPSUM)
            subsubcol1, subsubcol2 = st.columns(2)
            subsubcol1.write(LOREM_IPSUM)
            subsubcol2.write(LOREM_IPSUM)
        subcol2.write(LOREM_IPSUM)
        st.write(LOREM_IPSUM)

if st.button("Nested columns - in sidebar (raises exception)"):
    with st.sidebar:
        col1, col2 = st.columns(2)
        col1.text_input("Text input 1")
        col2.text_input("Text input 2")
        col3, col4 = col1.columns(2)
        col3.text_input("Text input 3")
        col4.text_input("Text input 4")
        st.text_input("Text input 5")
