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
    click_checkbox,
    click_toggle,
)


@pytest.mark.performance
def test_form_input_performance(app: Page):
    """
    Tests the re-render performance when typing in an input that is in a form.
    """
    form_1 = app.get_by_test_id("stForm").nth(0)
    form_1.get_by_test_id("stTextArea").locator("textarea").press_sequentially(
        "this is some text", delay=100
    )
    wait_for_app_run(app)


def change_widget_values(app: Page):
    """Change the checkbox value."""
    # Get the first form:
    form_1 = app.get_by_test_id("stForm").nth(0)
    click_checkbox(app, "Checkbox")

    # Change the date input value.
    form_1.get_by_test_id("stDateInput").locator("input").click()
    app.locator(
        '[data-baseweb="calendar"] [aria-label^="Choose Wednesday, July 17th 2019."]'
    ).first.click()

    # Change the multiselect value.
    form_1.get_by_test_id("stMultiSelect").locator("input").click()
    app.locator("[data-baseweb='popover'] >> li").nth(0).click()

    # Change the number input value.
    form_1.get_by_test_id("stNumberInput").locator("input").fill("42")

    # Change the radio value.
    form_1.get_by_test_id("stRadio").locator('label[data-baseweb="radio"]').nth(
        1
    ).click(force=True)

    # Change the selectbox value.
    form_1.get_by_test_id("stSelectbox").locator("input").click()
    app.locator("[data-baseweb='popover']").locator("li").nth(1).click()

    # Change the select slider value.
    form_1.get_by_test_id("stSlider").nth(0).get_by_role("slider").press("ArrowRight")

    # Change the slider value.
    form_1.get_by_test_id("stSlider").nth(1).get_by_role("slider").press("ArrowRight")

    # Change the text area value.
    form_1.get_by_test_id("stTextArea").locator("textarea").fill("bar")

    # Change the text input value.
    form_1.get_by_test_id("stTextInput").locator("input").fill("bar")

    # Change the time input value.
    form_1.get_by_test_id("stTimeInput").locator("input").click()
    app.locator('[data-baseweb="popover"]').locator("li").nth(0).click()

    # Change the toggle value.
    click_toggle(app, "Toggle Input")


def test_does_not_change_values_before_form_submitted(app: Page):
    """Query for markdown elements after the form."""
    markdown_elements = app.get_by_test_id("stMarkdown")

    # Change widget values without submitting the form.
    change_widget_values(app)

    # Assert that the values did not change.
    expect(markdown_elements.nth(0)).to_have_text("Checkbox: False")
    expect(markdown_elements.nth(1)).to_have_text("Date Input: 2019-07-06")
    expect(markdown_elements.nth(2)).to_have_text("Multiselect: foo")
    expect(markdown_elements.nth(3)).to_have_text("Number Input: 0.0")
    expect(markdown_elements.nth(4)).to_have_text("Radio: foo")
    expect(markdown_elements.nth(5)).to_have_text("Selectbox: foo")
    expect(markdown_elements.nth(6)).to_have_text("Select Slider: foo")
    expect(markdown_elements.nth(7)).to_have_text("Slider: 0")
    expect(markdown_elements.nth(8)).to_have_text("Text Area: foo")
    expect(markdown_elements.nth(9)).to_have_text("Text Input: foo")
    expect(markdown_elements.nth(10)).to_have_text("Time Input: 08:45:00")
    expect(markdown_elements.nth(11)).to_have_text("Toggle Input: False")


def test_changes_widget_values_after_form_submitted(app: Page):
    # Change widget values and submit the form.
    change_widget_values(app)
    app.get_by_test_id("stFormSubmitButton").nth(0).locator("button").click()
    wait_for_app_run(app)

    # Query for markdown elements after the form
    markdown_elements = app.get_by_test_id("stMarkdown")

    # Assert that the values have changed.
    expect(markdown_elements.nth(0)).to_have_text("Checkbox: True")
    expect(markdown_elements.nth(1)).to_have_text("Date Input: 2019-07-17")
    expect(markdown_elements.nth(2)).to_have_text("Multiselect: foo, bar")
    expect(markdown_elements.nth(3)).to_have_text("Number Input: 42.0")
    expect(markdown_elements.nth(4)).to_have_text("Radio: bar")
    expect(markdown_elements.nth(5)).to_have_text("Selectbox: bar")
    expect(markdown_elements.nth(6)).to_have_text("Select Slider: bar")
    expect(markdown_elements.nth(7)).to_have_text("Slider: 1")
    expect(markdown_elements.nth(8)).to_have_text("Text Area: bar")
    expect(markdown_elements.nth(9)).to_have_text("Text Input: bar")
    expect(markdown_elements.nth(10)).to_have_text("Time Input: 00:00:00")
    expect(markdown_elements.nth(11)).to_have_text("Toggle Input: True")


def test_form_with_stretched_button(
    themed_app: Page, assert_snapshot: ImageCompareFunction
):
    """Tests if the form with stretched submit button renders correctly."""
    form_2 = themed_app.get_by_test_id("stForm").nth(1)

    assert_snapshot(form_2, name="st_form-with_stretched_submit_button")

    submit_buttons = form_2.get_by_test_id("stFormSubmitButton")
    expect(submit_buttons).to_have_count(2)

    submit_button = submit_buttons.nth(0)
    submit_button.hover()
    expect(themed_app.get_by_test_id("stTooltipContent")).to_have_text(
        "Submit by clicking"
    )


def test_form_submit_with_emoji_icon(app: Page, assert_snapshot: ImageCompareFunction):
    """Tests if the form submit button with emoji icon renders correctly."""
    form_4 = app.get_by_test_id("stForm").nth(3)

    assert_snapshot(form_4, name="st_form_submit-emoji_icon")


def test_form_submit_with_material_icon(
    app: Page, assert_snapshot: ImageCompareFunction
):
    """Tests if the form submit button with material icon renders correctly."""
    form_5 = app.get_by_test_id("stForm").nth(4)

    assert_snapshot(form_5, name="st_form_submit-material_icon")


def test_form_submits_on_enter(app: Page):
    """Tests that submit on enter works when 1st submit button enabled."""
    form_6 = app.get_by_test_id("stForm").nth(5)
    text_input = form_6.get_by_test_id("stTextInput").locator("input")
    text_input.type("Test")
    expect(form_6.get_by_test_id("InputInstructions")).to_have_text(
        "Press Enter to submit form"
    )

    # Submit the form by pressing Enter, check if submitted.
    text_input.press("Enter")
    wait_for_app_run(app)
    expect(form_6.get_by_test_id("stMarkdown").last).to_have_text("Form submitted")


def test_form_disabled_submit_on_enter(app: Page):
    """Tests that submit on enter does not work when 1st submit button disabled."""
    form_7 = app.get_by_test_id("stForm").nth(6)
    text_input = form_7.get_by_test_id("stTextInput").locator("input")
    text_input.fill("Test")
    expect(form_7.get_by_test_id("InputInstructions")).to_have_text("")

    # Try to submit the form by pressing Enter, check not submitted.
    text_input.press("Enter")
    wait_for_app_run(app)
    expect(form_7.get_by_test_id("stMarkdown").last).not_to_have_text("Form submitted")


def test_enter_to_submit_false(app: Page):
    """Tests that pressing Enter does not submit form when enter_to_submit=False."""
    form_8 = app.get_by_test_id("stForm").nth(7)
    number_input = form_8.get_by_test_id("stNumberInput").locator("input")
    number_input.fill("42")
    expect(form_8.get_by_test_id("InputInstructions")).to_have_text("")

    # Try to submit the form by pressing Enter, check not submitted.
    number_input.press("Enter")
    wait_for_app_run(app)
    expect(form_8.get_by_test_id("stMarkdown").last).not_to_have_text("Form submitted")


def test_form_submits_on_click(app: Page):
    """Tests that submit via enabled form submit button works."""
    form_6 = app.get_by_test_id("stForm").nth(5)
    text_input = form_6.get_by_test_id("stTextInput").locator("input")
    text_input.fill("Test")
    expect(form_6.get_by_test_id("InputInstructions")).to_have_text(
        "Press Enter to submit form"
    )

    # Submit form with enabled submit button, check submitted
    form_6.get_by_test_id("stFormSubmitButton").first.click()
    wait_for_app_run(app)
    expect(form_6.get_by_test_id("stMarkdown").last).to_have_text("Form submitted")


def test_form_disabled_submit_on_click(app: Page):
    """Tests that submit via disabled form submit button does not work."""
    form_7 = app.get_by_test_id("stForm").nth(6)
    text_input = form_7.get_by_test_id("stTextInput").locator("input")
    text_input.fill("Test")
    expect(form_7.get_by_test_id("InputInstructions")).to_have_text("")

    # Try submit with disabled submit button, check not submitted
    form_7.get_by_test_id("stFormSubmitButton").first.click()
    wait_for_app_run(app)
    expect(form_7.get_by_test_id("stMarkdown").last).not_to_have_text("Form submitted")


def test_secondary_submit_buttons_enabled(app: Page):
    """Tests that secondary submit buttons work when enabled."""
    form_7 = app.get_by_test_id("stForm").nth(6)
    text_input = form_7.get_by_test_id("stTextInput").locator("input")
    text_input.fill("Test")
    expect(form_7.get_by_test_id("InputInstructions")).to_have_text("")

    # Submit form with secondary submit button, check submitted
    form_7.get_by_test_id("stFormSubmitButton").last.click()
    wait_for_app_run(app)
    expect(form_7.get_by_test_id("stMarkdown").last).to_have_text("Form submitted")


def test_borderless_form(app: Page, assert_snapshot: ImageCompareFunction):
    """Tests if the borderless form (border=False) renders correctly."""
    form_3 = app.get_by_test_id("stForm").nth(2)

    assert_snapshot(form_3, name="st_form-borderless")


def test_check_top_level_class(app: Page):
    """Check that the top level class is correctly set."""
    check_top_level_class(app, "stForm")


def test_check_form_submit_button_types(
    app: Page, assert_snapshot: ImageCompareFunction
):
    """Check that the form submit button types are correctly set."""
    form_9 = app.get_by_test_id("stForm").nth(8)
    assert_snapshot(form_9, name="st_form-primary_submit_button")

    form_10 = app.get_by_test_id("stForm").nth(9)
    assert_snapshot(form_10, name="st_form-tertiary_submit_button")
