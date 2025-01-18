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

"""Arrow marshalling unit tests."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pyarrow as pa

import streamlit as st
from streamlit.dataframe_util import (
    convert_arrow_bytes_to_pandas_df,
    convert_arrow_table_to_arrow_bytes,
)
from tests.delta_generator_test_case import DeltaGeneratorTestCase


def mock_data_frame():
    return pd.DataFrame(
        index=[[0, 1], ["i1", "i2"]],
        columns=[[2, 3, 4], ["c1", "c2", "c3"]],
        data=np.arange(0, 6, 1).reshape(2, 3),
    )


class ArrowTest(DeltaGeneratorTestCase):
    """Test ability to marshall arrow protos."""

    def test_dataframe_data(self):
        df = mock_data_frame()
        st.table(df)

        proto = self.get_delta_from_queue().new_element.arrow_table
        pd.testing.assert_frame_equal(convert_arrow_bytes_to_pandas_df(proto.data), df)

    def test_pyarrow_table_data(self):
        df = mock_data_frame()
        table = pa.Table.from_pandas(df)
        st.table(table)

        proto = self.get_delta_from_queue().new_element.arrow_table
        self.assertEqual(proto.data, convert_arrow_table_to_arrow_bytes(table))

    def test_uuid(self):
        df = mock_data_frame()
        styler = df.style
        styler.set_uuid("FAKE_UUID")
        st.table(styler)

        proto = self.get_delta_from_queue().new_element.arrow_table
        self.assertEqual(proto.styler.uuid, "FAKE_UUID")

    def test_caption(self):
        df = mock_data_frame()
        styler = df.style
        styler.set_caption("FAKE_CAPTION")
        st.table(styler)

        proto = self.get_delta_from_queue().new_element.arrow_table
        self.assertEqual(proto.styler.caption, "FAKE_CAPTION")

    def test_table_styles(self):
        df = mock_data_frame()
        styler = df.style
        # NOTE: If UUID is not set - a random UUID will be generated.
        styler.set_uuid("FAKE_UUID")
        styler.set_table_styles(
            [{"selector": ".blank", "props": [("background-color", "red")]}]
        )
        st.table(styler)

        proto = self.get_delta_from_queue().new_element.arrow_table
        self.assertEqual(
            proto.styler.styles, "#T_FAKE_UUID .blank { background-color: red }"
        )

    def test_cell_styles(self):
        df = mock_data_frame()
        styler = df.style
        # NOTE: If UUID is not set - a random UUID will be generated.
        styler.set_uuid("FAKE_UUID")
        styler.highlight_max(axis=None)
        st.table(styler)

        proto = self.get_delta_from_queue().new_element.arrow_table
        self.assertEqual(
            proto.styler.styles, "#T_FAKE_UUIDrow1_col2 { background-color: yellow }"
        )

    def test_display_values(self):
        df = pd.DataFrame(
            [[1, 2, 3], [4, 5, 6]],
        )
        styler = df.style.format("{:.2%}")
        st.table(styler)

        expected = pd.DataFrame(
            [["100.00%", "200.00%", "300.00%"], ["400.00%", "500.00%", "600.00%"]],
        )

        proto = self.get_delta_from_queue().new_element.arrow_table
        pd.testing.assert_frame_equal(
            convert_arrow_bytes_to_pandas_df(proto.styler.display_values), expected
        )

    def test_table_uses_convert_anything_to_df(self):
        """Test that st.table uses convert_anything_to_df to convert input data."""
        df = mock_data_frame()

        with patch(
            "streamlit.dataframe_util.convert_anything_to_pandas_df"
        ) as convert_anything_to_df:
            convert_anything_to_df.return_value = df

            st.table(df)
            convert_anything_to_df.assert_called_once()
