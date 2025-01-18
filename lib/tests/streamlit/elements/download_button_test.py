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

"""download_button unit test."""

from parameterized import parameterized

import streamlit as st
from tests.delta_generator_test_case import DeltaGeneratorTestCase


class DownloadButtonTest(DeltaGeneratorTestCase):
    """Test ability to marshall download_button protos."""

    @parameterized.expand([("hello world",), (b"byteshere",)])
    def test_just_label(self, data):
        """Test that it can be called with label and string or bytes data."""
        st.download_button("the label", data=data)

        c = self.get_delta_from_queue().new_element.download_button
        self.assertEqual(c.label, "the label")
        self.assertEqual(c.type, "secondary")
        self.assertEqual(c.disabled, False)

    def test_emoji_icon(self):
        """Test that it can be called with emoji icon."""
        st.download_button("the label", icon="⚡", data="juststring")

        c = self.get_delta_from_queue().new_element.download_button
        self.assertEqual(c.icon, "⚡")

    def test_material_icon(self):
        """Test that it can be called with material icon."""
        st.download_button("the label", icon=":material/thumb_up:", data="juststring")

        c = self.get_delta_from_queue().new_element.download_button
        self.assertEqual(c.icon, ":material/thumb_up:")

    def test_just_disabled(self):
        """Test that it can be called with disabled param."""
        st.download_button("the label", data="juststring", disabled=True)

        c = self.get_delta_from_queue().new_element.download_button
        self.assertEqual(c.disabled, True)

    def test_url_exist(self):
        """Test that file url exist in proto."""
        st.download_button("the label", data="juststring")

        c = self.get_delta_from_queue().new_element.download_button
        self.assertTrue("/media/" in c.url)

    @parameterized.expand(["primary", "secondary", "tertiary"])
    def test_type(self, type):
        """Test that it can be called with type param."""
        st.download_button("the label", data="Streamlit", type=type)

        c = self.get_delta_from_queue().new_element.download_button
        self.assertEqual(c.type, type)

    def test_use_container_width_can_be_set_to_true(self):
        """Test use_container_width can be set to true."""
        st.download_button("the label", data="juststring", use_container_width=True)

        c = self.get_delta_from_queue().new_element.download_button
        self.assertEqual(c.use_container_width, True)

    def test_use_container_width_is_false_by_default(self):
        """Test use_container_width is false by default."""
        st.download_button("the label", data="juststring")

        c = self.get_delta_from_queue().new_element.download_button
        self.assertEqual(c.use_container_width, False)

    def test_shows_cached_widget_replay_warning(self):
        """Test that a warning is shown when this widget is used inside a cached function."""
        st.cache_data(lambda: st.download_button("the label", data="juststring"))()

        # The widget itself is still created, so we need to go back one element more:
        el = self.get_delta_from_queue(-2).new_element.exception
        self.assertEqual(el.type, "CachedWidgetWarning")
        self.assertTrue(el.is_warning)
