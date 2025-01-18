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

from e2e_playwright.conftest import ImageCompareFunction
from e2e_playwright.shared.app_utils import check_top_level_class

TOTAL_BAR_CHARTS = 18


def test_bar_chart_rendering(app: Page, assert_snapshot: ImageCompareFunction):
    """Test that st.bar_chart renders correctly via snapshot testing."""
    bar_chart_elements = app.get_by_test_id("stVegaLiteChart")
    expect(bar_chart_elements).to_have_count(TOTAL_BAR_CHARTS)

    # Also make sure that all canvas objects are rendered:
    expect(bar_chart_elements.locator("canvas")).to_have_count(TOTAL_BAR_CHARTS)

    for i, element in enumerate(bar_chart_elements.all()):
        assert_snapshot(element, name=f"st_bar_chart-{i}")


def test_themed_bar_chart_rendering(
    themed_app: Page, assert_snapshot: ImageCompareFunction
):
    """Test that st.bar_chart renders with different theming."""
    bar_chart_elements = themed_app.get_by_test_id("stVegaLiteChart")
    expect(bar_chart_elements).to_have_count(TOTAL_BAR_CHARTS)

    # Also make sure that all canvas objects are rendered:
    expect(bar_chart_elements.locator("canvas")).to_have_count(TOTAL_BAR_CHARTS)

    # Only test a single chart per built-in chart type:
    assert_snapshot(bar_chart_elements.nth(1), name="st_bar_chart_themed")


def test_check_top_level_class(app: Page):
    """Check that the top level class is correctly set."""
    check_top_level_class(app, "stVegaLiteChart")
