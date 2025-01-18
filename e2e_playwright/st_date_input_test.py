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

from e2e_playwright.conftest import ImageCompareFunction, wait_for_app_run
from e2e_playwright.shared.app_utils import (
    check_top_level_class,
    expect_help_tooltip,
    get_element_by_key,
)


def test_date_input_rendering(themed_app: Page, assert_snapshot: ImageCompareFunction):
    """Test that st.date_input renders correctly via screenshots matching."""
    date_widgets = themed_app.get_by_test_id("stDateInput")
    expect(date_widgets).to_have_count(15)

    assert_snapshot(date_widgets.nth(0), name="st_date_input-single_date")
    assert_snapshot(date_widgets.nth(1), name="st_date_input-single_datetime")
    assert_snapshot(date_widgets.nth(2), name="st_date_input-range_no_date")
    assert_snapshot(date_widgets.nth(3), name="st_date_input-range_one_date")
    assert_snapshot(date_widgets.nth(4), name="st_date_input-range_two_dates")
    assert_snapshot(date_widgets.nth(5), name="st_date_input-disabled_no_date")
    assert_snapshot(date_widgets.nth(6), name="st_date_input-label_hidden")
    assert_snapshot(date_widgets.nth(7), name="st_date_input-label_collapsed")
    assert_snapshot(date_widgets.nth(8), name="st_date_input-single_date_format")
    assert_snapshot(date_widgets.nth(9), name="st_date_input-range_two_dates_format")
    assert_snapshot(date_widgets.nth(10), name="st_date_input-range_no_date_format")
    assert_snapshot(date_widgets.nth(11), name="st_date_input-single_date_callback")
    assert_snapshot(date_widgets.nth(12), name="st_date_input-empty_value")
    assert_snapshot(date_widgets.nth(13), name="st_date_input-value_from_state")
    assert_snapshot(date_widgets.nth(14), name="st_date_input-markdown_label")


def test_help_tooltip_works(app: Page):
    leading_indent_regular_text_tooltip = """
    This is a regular text block!
    Test1
    Test2

    """
    element_with_help = app.get_by_test_id("stDateInput").nth(0)
    expect_help_tooltip(app, element_with_help, leading_indent_regular_text_tooltip)


def test_date_input_has_correct_initial_values(app: Page):
    """Test that st.date_input has the correct initial values."""
    markdown_elements = app.get_by_test_id("stMarkdown")
    expect(markdown_elements).to_have_count(15)

    expected = [
        "Value 1: 1970-01-01",
        "Value 2: 2019-07-06",
        "Value 3: ()",
        "Value 4: (datetime.date(2019, 7, 6),)",
        "Value 5: (datetime.date(2019, 7, 6), datetime.date(2019, 7, 8))",
        "Value 6: ()",
        "Value 7: 2019-07-06",
        "Value 8: 2019-07-06",
        "Value 9: 1970-01-01",
        "Value 10: (datetime.date(2019, 7, 6), datetime.date(2019, 7, 8))",
        "Value 11: ()",
        "Value 12: 1970-01-01",
        "Date Input Changed: False",
        "Value 13: None",
        "Value 14: 1970-02-03",
    ]

    for markdown_element, expected_text in zip(markdown_elements.all(), expected):
        expect(markdown_element).to_have_text(expected_text, use_inner_text=True)


def test_handles_date_selection(app: Page):
    """Test that selection of a date on the calendar works as expected."""
    app.get_by_test_id("stDateInput").first.click()

    # Select '1970/01/02':
    app.locator(
        '[data-baseweb="calendar"] [aria-label^="Choose Friday, January 2nd 1970."]'
    ).first.click()

    expect(app.get_by_test_id("stMarkdown").first).to_have_text(
        "Value 1: 1970-01-02", use_inner_text=True
    )


def test_handle_value_changes(app: Page):
    """Test that st.date_input has the correct value after typing in a date."""

    first_date_input_field = app.get_by_test_id("stDateInput").first.locator("input")
    first_date_input_field.fill("1970/01/02")
    first_date_input_field.blur()

    expect(app.get_by_test_id("stMarkdown").first).to_have_text(
        "Value 1: 1970-01-02", use_inner_text=True
    )


def test_empty_date_input_behaves_correctly(
    app: Page, assert_snapshot: ImageCompareFunction
):
    """Test that st.date_input behaves correctly when empty."""
    # Enter 10 in the first empty input:
    empty_number_input = app.get_by_test_id("stDateInput").nth(12).locator("input")
    empty_number_input.type("1970/01/02", delay=50)
    empty_number_input.press("Enter")
    wait_for_app_run(app)
    expect(app.get_by_test_id("stMarkdown").nth(13)).to_have_text(
        "Value 13: 1970-01-02", use_inner_text=True
    )

    # Click outside to remove focus:
    app.get_by_test_id("stMarkdown").nth(13).click()

    # Screenshot match clearable input:
    assert_snapshot(
        app.get_by_test_id("stDateInput").nth(12),
        name="st_date_input-clearable_input",
        image_threshold=0.035,
    )

    # Press escape to clear value:
    empty_number_input = app.get_by_test_id("stDateInput").nth(12).locator("input")
    empty_number_input.focus()
    empty_number_input.press("Escape")
    # Click outside to enter value:
    app.get_by_test_id("stMarkdown").nth(13).click()

    # Should be empty again:
    expect(app.get_by_test_id("stMarkdown").nth(13)).to_have_text(
        "Value 13: None", use_inner_text=True
    )


def test_handles_range_end_date_changes(app: Page):
    """Test that it correctly handles changes to the end date of a range."""
    app.get_by_test_id("stDateInput").nth(3).click()

    # Select '2019/07/10'
    app.locator(
        '[data-baseweb="calendar"] [aria-label^="Choose Wednesday, July 10th 2019."]'
    ).first.click()

    expect(app.get_by_test_id("stMarkdown").nth(3)).to_have_text(
        "Value 4: (datetime.date(2019, 7, 6), datetime.date(2019, 7, 10))",
        use_inner_text=True,
    )


def test_handles_range_start_end_date_changes(app: Page):
    """Test that it correctly handles changes to the start and end date of a range."""
    app.get_by_test_id("stDateInput").nth(4).click()

    # Select start date: '2019/07/10'
    app.locator(
        '[data-baseweb="calendar"] [aria-label^="Choose Wednesday, July 10th 2019."]'
    ).first.click()

    expect(app.get_by_test_id("stMarkdown").nth(4)).to_have_text(
        "Value 5: (datetime.date(2019, 7, 10),)",
        use_inner_text=True,
    )

    # Select end date: '2019/07/12'
    app.locator(
        '[data-baseweb="calendar"] [aria-label^="Choose Friday, July 12th 2019."]'
    ).first.click()

    expect(app.get_by_test_id("stMarkdown").nth(4)).to_have_text(
        "Value 5: (datetime.date(2019, 7, 10), datetime.date(2019, 7, 12))",
        use_inner_text=True,
    )


def test_calls_callback_on_change(app: Page):
    """Test that it correctly calls the callback on change."""
    app.get_by_test_id("stDateInput").nth(11).click()

    # Select '1970/01/02'
    calendar = app.locator(
        '[data-baseweb="calendar"] [aria-label^="Choose Friday, January 2nd 1970."]'
    ).first
    expect(calendar).to_be_visible()
    calendar.click()
    wait_for_app_run(app)

    value_12_element = app.get_by_test_id("stMarkdown").nth(11)
    expect(value_12_element).to_have_text(
        "Value 12: 1970-01-02",
        use_inner_text=True,
    )
    expect(app.get_by_test_id("stMarkdown").nth(12)).to_have_text(
        "Date Input Changed: True",
        use_inner_text=True,
    )

    # Change different date input to trigger delta path change
    first_date_input_field = app.get_by_test_id("stDateInput").first.locator("input")
    first_date_input_field.fill("1971/01/03")
    wait_for_app_run(app)

    expect(app.get_by_test_id("stMarkdown").first).to_have_text(
        "Value 1: 1971-01-03", use_inner_text=True
    )

    # Test if value is still correct after delta path change
    expect(value_12_element).to_have_text(
        "Value 12: 1970-01-02",
        use_inner_text=True,
    )
    expect(app.get_by_test_id("stMarkdown").nth(12)).to_have_text(
        "Date Input Changed: False",
        use_inner_text=True,
    )


def test_single_date_calendar_picker_rendering(
    themed_app: Page, assert_snapshot: ImageCompareFunction
):
    """Test that the single value calendar picker renders correctly via screenshots matching."""
    themed_app.get_by_test_id("stDateInput").first.click()
    assert_snapshot(
        themed_app.locator('[data-baseweb="calendar"]').first,
        name="st_date_input-single_date_calendar",
    )


def test_range_date_calendar_picker_rendering(
    themed_app: Page, assert_snapshot: ImageCompareFunction
):
    """Test that the range calendar picker renders correctly via screenshots matching."""
    themed_app.get_by_test_id("stDateInput").nth(4).click()
    assert_snapshot(
        themed_app.locator('[data-baseweb="calendar"]').first,
        name="st_date_input-range_two_dates_calendar",
    )


def test_resets_to_default_single_value_if_calendar_closed_empty(app: Page):
    """Test that single value is reset to default if calendar closed empty."""
    app.get_by_test_id("stDateInput").first.click()

    # Select '1970/01/02'
    app.locator(
        '[data-baseweb="calendar"] [aria-label^="Choose Friday, January 2nd 1970."]'
    ).first.click()

    expect(app.get_by_test_id("stMarkdown").first).to_have_text(
        "Value 1: 1970-01-02", use_inner_text=True
    )

    # Close calendar without selecting a date
    date_input_field = app.get_by_test_id("stDateInput").first.locator("input")
    date_input_field.focus()
    date_input_field.clear()

    # Click outside of the calendar to submit value
    app.get_by_test_id("stMarkdown").first.click(delay=100)

    # Value should be reset to default
    expect(app.get_by_test_id("stMarkdown").first).to_have_text(
        "Value 1: 1970-01-01", use_inner_text=True
    )


def test_range_is_empty_if_calendar_closed_empty(app: Page):
    """Test that range value is empty of calendar closed empty."""
    app.get_by_test_id("stDateInput").nth(4).click()

    # Select start date: '2019/07/10'
    app.locator(
        '[data-baseweb="calendar"] [aria-label^="Choose Wednesday, July 10th 2019."]'
    ).first.click()

    expect(app.get_by_test_id("stMarkdown").nth(4)).to_have_text(
        "Value 5: (datetime.date(2019, 7, 10),)",
        use_inner_text=True,
    )

    # Select end date: '2019/07/12'
    app.locator(
        '[data-baseweb="calendar"] [aria-label^="Choose Friday, July 12th 2019."]'
    ).first.click()

    expect(app.get_by_test_id("stMarkdown").nth(4)).to_have_text(
        "Value 5: (datetime.date(2019, 7, 10), datetime.date(2019, 7, 12))",
        use_inner_text=True,
    )

    # Close calendar without selecting a date
    date_input_field = app.get_by_test_id("stDateInput").nth(4).locator("input")
    date_input_field.focus()
    date_input_field.clear()

    # Click outside of the calendar to submit value
    app.get_by_test_id("stMarkdown").nth(4).click()

    # Range should be empty
    expect(app.get_by_test_id("stMarkdown").nth(4)).to_have_text(
        "Value 5: ()",
        use_inner_text=True,
    )


def test_check_top_level_class(app: Page):
    """Check that the top level class is correctly set."""
    check_top_level_class(app, "stDateInput")


def test_custom_css_class_via_key(app: Page):
    """Test that the element can have a custom css class via the key argument."""
    expect(get_element_by_key(app, "date_input_12")).to_be_visible()
