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

"""Bokeh unit test."""

import unittest
from unittest.mock import patch

import numpy as np

import streamlit as st
from streamlit.errors import StreamlitAPIException
from streamlit.type_util import is_version_less_than
from tests.delta_generator_test_case import DeltaGeneratorTestCase


class BokehTest(DeltaGeneratorTestCase):
    """Test ability to marshall bokeh_chart protos."""

    @unittest.skipIf(
        is_version_less_than(np.__version__, "2.0.0") is False,
        "This test only runs if numpy is < 2.0.0. The bokeh version supported "
        "by Streamlit is not compatible with numpy 2.x.",
    )
    def test_figure(self):
        """Test that it can be called with figure."""

        from bokeh.plotting import figure

        plot = figure()
        plot.line([1], [1])
        st.bokeh_chart(plot)

        c = self.get_delta_from_queue().new_element.bokeh_chart
        self.assertEqual(hasattr(c, "figure"), True)

    @unittest.skipIf(
        is_version_less_than(np.__version__, "2.0.0") is False,
        "This test only runs if numpy is < 2.0.0. The bokeh version supported "
        "by Streamlit is not compatible with numpy 2.x.",
    )
    def test_bokeh_version_failure(self):
        from bokeh.plotting import figure

        with patch("bokeh.__version__", return_value="2.4.0"):
            plot = figure()
            with self.assertRaises(StreamlitAPIException):
                st.bokeh_chart(plot)
