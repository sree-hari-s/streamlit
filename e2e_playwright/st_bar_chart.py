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
from datetime import date

import numpy as np
import pandas as pd
from vega_datasets import data as vega_data

import streamlit as st

np.random.seed(0)


data = np.random.randn(20, 3)
df = pd.DataFrame(data, columns=["a", "b", "c"])

# st.area/bar/line_chart all use Altair/Vega-Lite under the hood.
# By default, Vega-Lite displays time values in the browser's local
# time zone, but data is sent down to the browser as UTC. This means
# Times need to be set correctly to the users timezone.
utc_df = pd.DataFrame(
    {
        "index": [
            date(2019, 8, 9),
            date(2019, 8, 10),
            date(2019, 8, 11),
            date(2019, 8, 12),
        ],
        "numbers": [10, 50, 30, 40],
    }
)

utc_df.set_index("index", inplace=True)

# Dataframe to test the color parameter support:
N = 100

color_df = pd.DataFrame(
    {
        # Using a negative range so certain kinds of bugs are more visible.
        "a": -np.arange(N),
        "b": np.random.rand(N) * 10,
        "c": np.random.rand(N) * 10,
        "d": np.random.randn(N) * 30,
        "e": ["bird" if x % 2 else "airplane" for x in range(N)],
    }
)

st.header("Bar Chart")

st.bar_chart()
st.bar_chart(df)
st.bar_chart(df, x="a")
st.bar_chart(df, y="a")
st.bar_chart(df, y=["a", "b"])
st.bar_chart(df, x="a", y="b", height=500, width=300, use_container_width=False)
st.bar_chart(df, x="b", y="a")
st.bar_chart(df, x="a", y=["b", "c"])
st.bar_chart(utc_df)
st.bar_chart(color_df, x="a", y="b", color="e")
st.bar_chart(df, x_label="X Axis Label", y_label="Y Axis Label")
st.bar_chart(df, horizontal=True)
st.bar_chart(df, horizontal=True, x_label="X Label", y_label="Y Label")

# Additional tests for stacking options
source = vega_data.barley()
st.bar_chart(source, x="variety", y="yield", color="site", stack=True)
st.bar_chart(source, x="variety", y="yield", color="site", stack=False)
st.bar_chart(source, x="variety", y="yield", color="site", stack="normalize")
st.bar_chart(source, x="variety", y="yield", color="site", stack="center")
st.bar_chart(source, x="variety", y="yield", color="site", stack="layered")
