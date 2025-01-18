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

from e2e_playwright.conftest import ImageCompareFunction, wait_for_app_run
from e2e_playwright.shared.app_utils import (
    check_top_level_class,
    click_form_button,
    expect_help_tooltip,
    get_element_by_key,
)


def test_color_picker_widget_display(
    themed_app: Page, assert_snapshot: ImageCompareFunction
):
    """Test that st.color_picker renders correctly."""
    color_pickers = themed_app.get_by_test_id("stColorPicker")
    expect(color_pickers).to_have_count(8)
    assert_snapshot(color_pickers.nth(0), name="st_color_picker-regular")
    assert_snapshot(color_pickers.nth(1), name="st_color_picker-default_help")
    assert_snapshot(color_pickers.nth(2), name="st_color_picker-disabled")
    assert_snapshot(color_pickers.nth(3), name="st_color_picker-hidden_label")
    assert_snapshot(color_pickers.nth(4), name="st_color_picker-collapsed_label")
    # The other color pickers do not need to be snapshot tested since they
    # don't have any visually interesting differences.
    assert_snapshot(color_pickers.nth(7), name="st_color_picker-markdown_label")


def test_color_picker_popover_display(
    themed_app: Page, assert_snapshot: ImageCompareFunction
):
    """Test that color picker popover renders correctly in both themes."""
    color_pickers = themed_app.get_by_test_id("stColorPicker")
    color_pickers.nth(0).get_by_test_id("stColorPickerBlock").click()

    popover = themed_app.get_by_test_id("stColorPickerPopover")
    expect(popover).to_be_visible()
    assert_snapshot(popover, name="st_color_picker-popover")


def test_help_tooltip_works(app: Page):
    element_with_help = app.get_by_test_id("stColorPicker").nth(1)
    expect_help_tooltip(app, element_with_help, "help string")


# The coordinates (0, 0) for the click action behaves differently across firefox.
@pytest.mark.skip_browser("firefox")
def test_clicking_color_on_color_picker_works(
    app: Page, assert_snapshot: ImageCompareFunction
):
    color_pickers = app.get_by_test_id("stColorPicker")
    color_pickers.nth(0).get_by_test_id("stColorPickerBlock").click()

    app.get_by_test_id("stColorPickerPopover").click(position={"x": 0, "y": 0})

    # click outside of color picker
    app.get_by_text("Default Color").click()
    wait_for_app_run(app)
    expect(app.get_by_text("#ffffff")).to_be_visible()
    assert_snapshot(color_pickers.nth(0), name="st_color_picker-clicked_new_color")


def test_typing_new_hex_color_on_color_picker_works_with_callback(
    app: Page, assert_snapshot: ImageCompareFunction
):
    expect(app.get_by_text("Hello world")).to_have_count(0)
    color_pickers = app.get_by_test_id("stColorPicker")
    color_pickers.nth(0).get_by_test_id("stColorPickerBlock").click()

    text_input = app.get_by_test_id("stColorPickerPopover").locator("input")
    text_input.fill("#ffffff")

    # click outside of color picker
    app.get_by_text("Default Color").click()
    wait_for_app_run(app)

    # callback writes "Hello world"
    expect(app.get_by_text("Hello world")).to_be_visible()
    expect(app.get_by_text("#ffffff")).to_be_visible()
    assert_snapshot(color_pickers.nth(0), name="st_color_picker-typed_new_hex_color")


def test_typing_new_RGB_color_on_color_picker_works(
    app: Page, assert_snapshot: ImageCompareFunction
):
    color_pickers = app.get_by_test_id("stColorPicker")
    color_pickers.nth(0).get_by_test_id("stColorPickerBlock").click()

    color_picker_popover = app.get_by_test_id("stColorPickerPopover")

    # click button to swap color picker mode to RGB
    color_picker_popover.locator("svg").click()

    rgb_text_inputs = app.get_by_test_id("stColorPickerPopover").locator("input")
    rgb_text_inputs.nth(0).type("255")
    rgb_text_inputs.nth(1).type("255")
    rgb_text_inputs.nth(2).type("255")

    # click outside of color picker
    app.get_by_text("Default Color").click()
    wait_for_app_run(app)
    expect(app.get_by_text("#ffffff")).to_be_visible()
    assert_snapshot(color_pickers.nth(0), name="st_color_picker-typed_new_rgb_color")


def test_typing_new_HSL_color_on_color_picker_works(
    app: Page, assert_snapshot: ImageCompareFunction
):
    color_pickers = app.get_by_test_id("stColorPicker")
    color_pickers.nth(0).get_by_test_id("stColorPickerBlock").click()

    color_picker_popover = app.get_by_test_id("stColorPickerPopover")

    # click button to swap color picker mode to HSL
    color_picker_input_button = color_picker_popover.locator("svg")
    color_picker_input_button.click()
    color_picker_input_button.click()

    hsl_text_inputs = app.get_by_test_id("stColorPickerPopover").locator("input")
    hsl_text_inputs.nth(0).fill("0%")
    hsl_text_inputs.nth(1).fill("0%")
    hsl_text_inputs.nth(2).fill("100%")

    # click outside of color picker
    app.get_by_text("Default Color").click()
    wait_for_app_run(app)
    expect(app.get_by_text("#ffffff")).to_be_visible()
    assert_snapshot(color_pickers.nth(0), name="st_color_picker-typed_new_hsl_color")


def test_in_form_selection_and_session_state(app: Page):
    expect(app.get_by_text("color_picker-in-form selection: #000000")).to_be_visible()
    expect(
        app.get_by_text("color_picker-in-form selection in session state: #000000")
    ).to_be_visible()

    app.get_by_test_id("stColorPicker").nth(5).get_by_test_id(
        "stColorPickerBlock"
    ).click()

    text_input = app.get_by_test_id("stColorPickerPopover").locator("input")
    text_input.fill("#ffffff")

    # click outside of color picker
    app.get_by_text("Default Color").click()
    wait_for_app_run(app)

    click_form_button(app, "Submit")

    expect(app.get_by_text("color_picker-in-form selection: #ffffff")).to_be_visible()
    expect(
        app.get_by_text("color_picker-in-form selection in session state: #ffffff")
    ).to_be_visible()


def test_color_picker_in_fragment(app: Page):
    expect(
        app.get_by_text("color_picker-in-fragment selection: #000000")
    ).to_be_visible()

    app.get_by_test_id("stColorPicker").nth(6).get_by_test_id(
        "stColorPickerBlock"
    ).click()
    text_input = app.get_by_test_id("stColorPickerPopover").locator("input")
    text_input.fill("#ffffff")

    # click outside of color picker
    app.get_by_text("Default Color").click()

    wait_for_app_run(app)

    expect(
        app.get_by_text("color_picker-in-fragment selection: #ffffff")
    ).to_be_visible()
    expect(app.get_by_text("Runs: 1")).to_be_visible()


def test_check_top_level_class(app: Page):
    """Check that the top level class is correctly set."""
    check_top_level_class(app, "stColorPicker")


def test_custom_css_class_via_key(app: Page):
    """Test that the element can have a custom css class via the key argument."""
    expect(get_element_by_key(app, "color_picker_1")).to_be_visible()
