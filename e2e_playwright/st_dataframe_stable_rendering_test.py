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


from playwright.sync_api import Page, expect

from e2e_playwright.conftest import wait_for_app_loaded
from e2e_playwright.shared.app_utils import click_toggle


def test_dataframe_renders_without_crashing(app: Page):
    """Test that st.dataframe renders without crashing."""

    # Reload the page a couple of times to make sure that the dataframe
    # crash doesn't appear.
    # This test is safeguarding against potential regressions that
    # cause crashes as report in: https://github.com/streamlit/streamlit/issues/7949
    # But these crashes are usually more random, thats why we run
    # it for a couple of page reloads.
    # Also, even if there are crashes, its not gurunteed that they will
    # happen in our CI environment.
    for _ in range(5):
        dataframe_elements = app.get_by_test_id("stDataFrame")
        expect(dataframe_elements).to_have_count(7)
        expect(app.get_by_test_id("stAlertContainer")).not_to_be_attached()

        # Set use_container_width to False:
        click_toggle(app, "use_container_width")
        dataframe_elements = app.get_by_test_id("stDataFrame")
        expect(dataframe_elements).to_have_count(7)
        expect(app.get_by_test_id("stAlertContainer")).not_to_be_attached()

        # Reload the page:
        app.reload()
        wait_for_app_loaded(app)
