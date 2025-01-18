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

import pytest
from playwright.sync_api import Page, expect

test_dicts = [
    {"x": ["y"]},
    {"x": ["y"], "a": ["b"]},
    {"x": ["y", "1", "2.34"]},
    {"x": [""]},
]


@pytest.mark.parametrize("app_with_query_params", test_dicts, indirect=True)
def test_app_with_experimental_get_query_params(app_with_query_params: Page):
    page, test_dict = app_with_query_params
    expect(page.get_by_test_id("stMarkdownContainer").nth(0)).to_contain_text(
        "Please replace st.experimental_get_query_params with st.query_params. "
        "st.experimental_get_query_params will be removed after 2024-04-11. "
        "Refer to our docs page for more information."
    )
    expect(page.get_by_test_id("stMarkdownContainer").nth(1)).to_contain_text(
        str(test_dict)
    )
