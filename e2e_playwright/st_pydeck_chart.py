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

import math
from typing import Any, cast

import numpy as np
import pandas as pd
import pydeck as pdk

import streamlit as st

# Empty chart.

st.pydeck_chart()

# Basic chart.

np.random.seed(12345)

df = pd.DataFrame(
    cast(Any, np.random.randn(1000, 2) / [50, 50]) + [37.76, -122.4],
    columns=["lat", "lon"],
)

st.pydeck_chart(
    pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v9",
        initial_view_state=pdk.ViewState(
            latitude=37.76,
            longitude=-122.4,
            zoom=11,
            pitch=50,
        ),
        layers=[
            pdk.Layer(
                "HexagonLayer",
                data=df,
                get_position="[lon, lat]",
                radius=200,
                elevation_scale=4,
                elevation_range=[0, 1000],
                pickable=True,
                extruded=True,
            ),
            pdk.Layer(
                "ScatterplotLayer",
                data=df,
                get_position="[lon, lat]",
                get_color="[200, 30, 0, 160]",
                get_radius=200,
            ),
        ],
    )
)

# Chart w/ invalid JSON - issue #5799.
data = pd.DataFrame({"lng": [-109.037673], "lat": [36.994672], "weight": [math.nan]})
layer = pdk.Layer(
    "ScatterplotLayer", data=data, get_position=["lng", "lat"], radius_min_pixels=4
)
deck = pdk.Deck(
    layers=[layer],
    map_style=pdk.map_styles.CARTO_LIGHT,
    tooltip={"text": "weight: {weight}"},
)
st.pydeck_chart(deck, use_container_width=True)

H3_HEX_DATA = [
    {"hex": "88283082b9fffff", "count": 10},
    {"hex": "88283082d7fffff", "count": 50},
    {"hex": "88283082a9fffff", "count": 100},
]
df = pd.DataFrame(H3_HEX_DATA)

st.pydeck_chart(
    pdk.Deck(
        map_style="mapbox://styles/mapbox/outdoors-v12",
        tooltip={"text": "Count: {count}"},
        initial_view_state=pdk.ViewState(
            latitude=37.7749295, longitude=-122.4194155, zoom=12, bearing=0, pitch=30
        ),
        layers=[
            pdk.Layer(
                "H3HexagonLayer",
                df,
                pickable=True,
                stroked=True,
                filled=True,
                get_hexagon="hex",
                get_fill_color="[0, 255, 0]",
                get_line_color=[255, 255, 255],
                line_width_min_pixels=2,
            ),
        ],
    )
)

st.pydeck_chart(
    pdk.Deck(
        initial_view_state=pdk.ViewState(
            latitude=37.76,
            longitude=-122.4,
            zoom=11,
            pitch=50,
        ),
        layers=[
            pdk.Layer(
                "HexagonLayer",
                data=df,
                get_position="[lon, lat]",
                radius=200,
                elevation_scale=4,
                elevation_range=[0, 1000],
                pickable=True,
                extruded=True,
            ),
            pdk.Layer(
                "ScatterplotLayer",
                data=df,
                get_position="[lon, lat]",
                get_color="[200, 30, 0, 160]",
                get_radius=200,
            ),
        ],
    )
)

st.pydeck_chart(
    pdk.Deck(
        map_style="mapbox://styles/mapbox/outdoors-v12",
        initial_view_state=pdk.ViewState(
            latitude=37.7749295, longitude=-122.4194155, zoom=12, bearing=0, pitch=30
        ),
        layers=[
            pdk.Layer(
                "H3HexagonLayer",
                df,
                pickable=True,
                stroked=True,
                filled=True,
                get_hexagon="hex",
                get_fill_color="[0, 255, 0]",
                get_line_color=[255, 255, 255],
                line_width_min_pixels=2,
            ),
        ],
    ),
    width=200,
    height=250,
    use_container_width=False,
)
