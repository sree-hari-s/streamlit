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

import time

import streamlit as st

st.header("Widget State - Heavy Usage Test")
# Test for https://github.com/streamlit/streamlit/issues/4836

number = st.number_input("test", value=0, step=1)
st.write(number)

if number:
    time.sleep(1)

st.header("Widget State - Redisplayed Widget Test")
# Test for https://github.com/streamlit/streamlit/issues/3512

if st.checkbox("Display widgets"):
    if st.checkbox("Show hello"):
        st.write("hello")

    if st.checkbox("Show goodbye", key="c3"):
        st.write("goodbye")

st.header("Test for input change & button click in one motion")
# Test for https://github.com/streamlit/streamlit/issues/10007


def btn_callback():
    st.write("Input: " + st.session_state["key1"])


st.text_area("Type something into the text area", key="key1")
st.button("Submit text_area", on_click=btn_callback)
