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

from __future__ import annotations

import pytest
from playwright.sync_api import FrameLocator, Locator, Page, Route, expect

from e2e_playwright.conftest import IframedPage, ImageCompareFunction, wait_for_app_run
from e2e_playwright.shared.app_utils import expect_prefixed_markdown, get_element_by_key
from e2e_playwright.shared.dataframe_utils import (
    calc_middle_cell_position,
    click_on_cell,
    expect_canvas_to_be_visible,
    get_open_cell_overlay,
)
from e2e_playwright.shared.toolbar_utils import (
    assert_fullscreen_toolbar_button_interactions,
)

# This test suite covers all interactions of dataframe & data_editor


def test_dataframe_toolbar_on_hover(
    themed_app: Page, assert_snapshot: ImageCompareFunction
):
    """Test that the toolbar is shown when hovering over a dataframe."""
    dataframe_element = themed_app.get_by_test_id("stDataFrame").nth(0)
    dataframe_toolbar = dataframe_element.get_by_test_id("stElementToolbar")

    # Check that it is currently not visible:
    expect(dataframe_toolbar).to_have_css("opacity", "0")

    # Hover over dataframe
    dataframe_element.hover()

    # Check that it is visible
    expect(dataframe_toolbar).to_have_css("opacity", "1")

    # Take a snapshot
    assert_snapshot(dataframe_toolbar, name="st_dataframe-toolbar")


def test_data_editor_toolbar_on_hover(
    themed_app: Page, assert_snapshot: ImageCompareFunction
):
    """Test that the toolbar is shown when hovering over a data editor component."""
    data_editor_element = themed_app.get_by_test_id("stDataFrame").nth(1)
    data_editor_toolbar = data_editor_element.get_by_test_id("stElementToolbar")

    # Check that it is currently not visible:
    expect(data_editor_toolbar).to_have_css("opacity", "0")

    # Hover over data editor:
    data_editor_element.hover()

    # Check that it is visible
    expect(data_editor_toolbar).to_have_css("opacity", "1")

    # Take a snapshot
    assert_snapshot(data_editor_toolbar, name="st_data_editor-toolbar")


def test_data_editor_delete_row_via_toolbar(
    themed_app: Page, assert_snapshot: ImageCompareFunction
):
    """Test that a row can be deleted via the toolbar."""
    data_editor_element = themed_app.get_by_test_id("stDataFrame").nth(1)
    data_editor_toolbar = data_editor_element.get_by_test_id("stElementToolbar")

    expect_canvas_to_be_visible(data_editor_element)
    # Select the second row
    data_editor_element.click(position={"x": 10, "y": 100})
    # Take a snapshot to check if row is selected:
    assert_snapshot(
        data_editor_element, name="st_data_editor-selected_row_for_deletion"
    )
    expect(data_editor_element).to_have_css("height", "247px")

    # The toolbar should be locked (visible):
    expect(data_editor_toolbar).to_have_css("opacity", "1")
    # Take snapshot to check if trash icon is in toolbar:
    assert_snapshot(data_editor_toolbar, name="st_data_editor-row_deletion_toolbar")

    # Click row deletion button:
    delete_row_button = data_editor_toolbar.get_by_test_id(
        "stElementToolbarButton"
    ).nth(0)
    delete_row_button.click()
    # The height should reflect that one row is missing (247px-35px=212px):
    expect(data_editor_element).to_have_css("height", "212px")


def test_data_editor_delete_row_via_hotkey(app: Page):
    """Test that a row can be deleted via delete hotkey."""
    data_editor_element = app.get_by_test_id("stDataFrame").nth(1)
    expect(data_editor_element).to_have_css("height", "247px")

    # Select the second row
    data_editor_element.click(position={"x": 10, "y": 100})

    # Press backspace to delete row:
    data_editor_element.press("Delete")

    # The height should reflect that one row is missing (247px-35px=212px):
    expect(data_editor_element).to_have_css("height", "212px")


def test_data_editor_add_row_via_toolbar(app: Page):
    """Test that a row can be added via the toolbar."""
    data_editor_element = app.get_by_test_id("stDataFrame").nth(1)
    data_editor_toolbar = data_editor_element.get_by_test_id("stElementToolbar")
    expect(data_editor_element).to_have_css("height", "247px")

    # Activate toolbar:
    data_editor_element.hover()
    # Check that it is visible
    expect(data_editor_toolbar).to_have_css("opacity", "1")

    # Click add row button:
    add_row_button = data_editor_toolbar.get_by_test_id("stElementToolbarButton").nth(0)
    add_row_button.click()

    # The height should reflect that one row is added (247px+35px=282px):
    expect(data_editor_element).to_have_css("height", "282px")


def test_data_editor_add_row_via_trailing_row(app: Page):
    """Test that a row can be added by clicking on the trailing row."""
    data_editor_element = app.get_by_test_id("stDataFrame").nth(1)
    expect(data_editor_element).to_have_css("height", "247px")

    # Click on the trailing row:
    data_editor_element.click(position={"x": 40, "y": 220})

    # The height should reflect that one row is added (247px+35px=282px):
    expect(data_editor_element).to_have_css("height", "282px")


# Firefox seems to be unable to run this test. But I tested it manually
# to make sure that it works correctly.
@pytest.mark.skip_browser("firefox")
def test_dataframe_toolbar_on_toolbar_hover(app: Page):
    """Test that the toolbar is shown when hovering over the toolbar."""
    dataframe_element = app.get_by_test_id("stDataFrame").nth(0)
    dataframe_toolbar = dataframe_element.get_by_test_id("stElementToolbar")

    # Check that it is currently not visible:
    expect(dataframe_toolbar).to_have_css("opacity", "0")

    # Hover over dataframe toolbar itself (which is position)
    dataframe_toolbar.hover(force=True, position={"x": 0, "y": 0})

    # Check that it is visible
    expect(dataframe_toolbar).to_have_css("opacity", "1")


def test_open_search_via_toolbar(
    themed_app: Page, assert_snapshot: ImageCompareFunction
):
    """Test that clicking on search toolbar button triggers dataframe search."""
    dataframe_element = themed_app.get_by_test_id("stDataFrame").nth(0)
    dataframe_toolbar = dataframe_element.get_by_test_id("stElementToolbar")
    search_toolbar_button = dataframe_toolbar.get_by_test_id(
        "stElementToolbarButton"
    ).nth(1)

    # Activate toolbar:
    dataframe_element.hover()
    # Check that it is visible
    expect(dataframe_toolbar).to_have_css("opacity", "1")

    # Hover search icon:
    search_toolbar_button.hover()
    # Test if tooltip works:
    expect(themed_app.get_by_test_id("stTooltipContent")).to_have_text("Search")
    # Take a screenshot to capture hover effect:
    assert_snapshot(dataframe_toolbar, name="st_dataframe-toolbar_hover_search")

    # Click on search button:
    search_toolbar_button.click()

    expect(themed_app.locator(".gdg-search-bar-inner")).to_be_visible()

    # Check that it is visible
    assert_snapshot(dataframe_element, name="st_dataframe-trigger_search_via_toolbar")


def test_open_search_via_hotkey(app: Page):
    """Test that the search can be opened via a hotkey."""
    dataframe_element = app.get_by_test_id("stDataFrame").nth(0)

    # Select a cell to focus the dataframe:
    click_on_cell(dataframe_element, 2, 3)

    # Press hotkey to open search:
    dataframe_element.press("Control+f")

    expect(app.locator(".gdg-search-bar-inner")).to_be_visible()


def test_clicking_on_fullscreen_toolbar_button(
    app: Page, assert_snapshot: ImageCompareFunction
):
    """Test that clicking on fullscreen toolbar button expands the dataframe into
    fullscreen."""

    assert_fullscreen_toolbar_button_interactions(
        app,
        assert_snapshot=assert_snapshot,
        widget_test_id="stDataFrame",
        filename_prefix="st_dataframe",
        nth=4,
    )


def test_data_editor_keeps_state_after_unmounting(
    app: Page, assert_snapshot: ImageCompareFunction
):
    """Test that the data editor keeps state correctly after unmounting."""
    data_editor_element = app.get_by_test_id("stDataFrame").nth(1)
    data_editor_toolbar = data_editor_element.get_by_test_id("stElementToolbar")
    expect(data_editor_element).to_have_css("height", "247px")

    # Activate toolbar:
    data_editor_element.hover()
    # Check that it is visible
    expect(data_editor_toolbar).to_have_css("opacity", "1")

    # Click add row button:
    add_row_button = data_editor_toolbar.get_by_test_id("stElementToolbarButton").nth(0)
    add_row_button.click()

    # The height should reflect that one row is added (247px+35px=282px):
    expect(data_editor_element).to_have_css("height", "282px")
    # The added row will trigger a rerun after a bounce, so we need to wait
    # for the app to finish running before we unmount the component.
    wait_for_app_run(app, 500)

    # Click button to unmount the component:
    app.get_by_test_id("stButton").locator("button").click()
    wait_for_app_run(app, 4000)

    # Check the height again, the row should be still attached:
    expect(data_editor_element).to_have_css("height", "282px")

    # Take a screenshot after unmounting:
    assert_snapshot(
        data_editor_element,
        name="st_data_editor-after_unmounting",
    )


def _test_csv_download(
    page: Page,
    locator: FrameLocator | Locator,
    click_enter_on_file_picker: bool = False,
):
    dataframe_element = locator.get_by_test_id("stDataFrame").nth(0)
    dataframe_toolbar = dataframe_element.get_by_test_id("stElementToolbar")

    download_csv_toolbar_button = dataframe_toolbar.get_by_test_id(
        "stElementToolbarButton"
    ).first

    # Activate toolbar:
    dataframe_element.hover()
    # Check that it is visible
    expect(dataframe_toolbar).to_have_css("opacity", "1")

    with page.expect_download(timeout=10000) as download_info:
        download_csv_toolbar_button.click()

        # playwright does not support all fileaccess APIs yet (see this
        # issue: https://github.com/microsoft/playwright/issues/8850) This means we
        # don't know if the system dialog opened to pick a location (expect_file_chooser
        # does not work). So as a workaround, we wait for now and then press enter.
        if click_enter_on_file_picker:
            page.wait_for_timeout(1000)
            page.keyboard.press("Enter")

    download = download_info.value
    download_path = download.path()
    with open(download_path, encoding="UTF-8") as f:
        content = f.read()
        # the app uses a fixed seed, so the data is always the same. This is the reason
        # why we can check it here.
        some_row = (
            "1,-0.977277879876411,0.9500884175255894,-0.1513572082976979,"
            "-0.10321885179355784,0.41059850193837233"
        )
        # we usually try to avoid assert in playwright tests, but since we don't have to
        # wait for any UI interaction or DOM state, it's ok here
        assert some_row in content


def test_csv_download_button(
    app: Page, browser_name: str, browser_type_launch_args: dict
):
    """Test that the csv download button works.

    Note that the library we are using calls the file picker API to download the file.
    This is not supported in headless mode. Hence, the test triggers different code
    paths in the app depending on the browser and the launch arguments.
    """

    click_enter_on_file_picker = False

    # right now the filechooser will only be opened on Chrome. Maybe this will change in
    # the future and the check has to be updated; or maybe playwright will support the
    # file-access APIs better. In headless mode, the file-access API our csv-download
    # button uses under-the-hood does not work. So we monkey-patch it to throw an error
    # and trigger our alternative download logic.
    if browser_name == "chromium":
        if browser_type_launch_args.get("headless", False):
            click_enter_on_file_picker = True
        else:
            app.evaluate(
                """() => window.showSaveFilePicker = () => {
                    throw new Error('Monkey-patched showOpenFilePicker')
                }""",
            )
    _test_csv_download(app, app.locator("body"), click_enter_on_file_picker)


def test_csv_download_button_in_iframe(iframed_app: IframedPage):
    """Test that the csv download button works in an iframe.

    Based on the test behavior and the fact that we don't have to patch the
    'window.showSaveFilePicker' as in the test above, it seems that the fallback
    download method is used.
    """

    page: Page = iframed_app.page
    frame_locator: FrameLocator = iframed_app.open_app(None)

    _test_csv_download(page, frame_locator)


def test_csv_download_button_in_iframe_with_new_tab_host_config(
    iframed_app: IframedPage,
):
    """Test that the csv download button works in an iframe and the host-config enforced
    download in new tab.

    Based on the test behavior and the fact that we don't have to patch the
    'window.showSaveFilePicker' as in the test above,
    it seems that the fallback download method is used.
    If this ever changes, the host-config[enforceDownloadInNewTab] might not take any
    effect as it is only used in the fallback mechanism.
    """
    page: Page = iframed_app.page

    def fulfill_host_config_request(route: Route):
        response = route.fetch()
        result = response.json()
        result["enforceDownloadInNewTab"] = True
        route.fulfill(json=result)

    page.route("**/_stcore/host-config", fulfill_host_config_request)

    # ensure that the route interception works and we get the correct
    # enforceDownloadInNewTab config
    with page.expect_event(
        "response",
        lambda response: response.url.endswith("_stcore/host-config")
        and response.json()["enforceDownloadInNewTab"] is True,
        timeout=10000,
    ):
        frame_locator: FrameLocator = iframed_app.open_app(None)
        _test_csv_download(page, frame_locator)


def test_number_cell_read_only_overlay_formatting(
    themed_app: Page, assert_snapshot: ImageCompareFunction
):
    """Test that the number cell overlay is formatted correctly."""
    overlay_test_df = themed_app.get_by_test_id("stDataFrame").nth(2)
    expect_canvas_to_be_visible(overlay_test_df)
    # Click on the first cell of the table
    click_on_cell(overlay_test_df, 1, 0, double_click=True, column_width="medium")
    cell_overlay = get_open_cell_overlay(themed_app)
    # Get the (number) input element and check the value
    expect(cell_overlay.locator(".gdg-input")).to_have_attribute("value", "1231231.41")
    assert_snapshot(cell_overlay, name="st_dataframe-number_col_overlay")


def _test_number_cell_editing(
    themed_app: Page,
    assert_snapshot: ImageCompareFunction,
    *,
    skip_snapshot: bool = False,
):
    """Test that the number cell can be edited."""
    cell_overlay_test_df = themed_app.get_by_test_id("stDataFrame").nth(3)
    expect_canvas_to_be_visible(cell_overlay_test_df)

    # Click on the first cell of the table
    click_on_cell(cell_overlay_test_df, 1, 0, double_click=True, column_width="medium")
    cell_overlay = get_open_cell_overlay(themed_app)
    # On some browsers the cell content is highlighted, so we enforce it to make the
    # test consistent and stable across all browsers
    cell_overlay.click()
    cell_overlay.press("ControlOrMeta+A")

    # Get the (number) input element and check the value
    expect(cell_overlay.locator(".gdg-input")).to_have_attribute("value", "1231231.41")
    if not skip_snapshot:
        assert_snapshot(cell_overlay, name="st_data_editor-number_col_editor")

    # Change the value
    cell_overlay.locator(".gdg-input").fill("9876.54")
    # Press Enter to apply the change
    themed_app.keyboard.press("Enter")
    wait_for_app_run(themed_app)

    # Check if that the value was submitted
    expect_prefixed_markdown(themed_app, "Edited DF:", "9876.54", exact_match=False)


def test_number_cell_editing(themed_app: Page, assert_snapshot: ImageCompareFunction):
    _test_number_cell_editing(themed_app, assert_snapshot)


@pytest.mark.performance
def test_number_cell_editing_performance(
    app: Page, assert_snapshot: ImageCompareFunction
):
    """Test that the number cell can be edited."""
    _test_number_cell_editing(app, assert_snapshot, skip_snapshot=True)


def test_text_cell_read_only_overlay_formatting(
    themed_app: Page, assert_snapshot: ImageCompareFunction
):
    """Test that the text cell overlay is formatted correctly."""
    overlay_test_df = themed_app.get_by_test_id("stDataFrame").nth(2)
    expect_canvas_to_be_visible(overlay_test_df)

    # Click on the first cell of the table
    click_on_cell(overlay_test_df, 1, 1, double_click=True, column_width="medium")
    cell_overlay = get_open_cell_overlay(themed_app)

    # Get the (text) input element and check the value
    expect(cell_overlay.locator(".gdg-input")).to_have_text("hello\nworld")
    assert_snapshot(cell_overlay, name="st_dataframe-text_col_overlay")


def test_text_cell_editing(themed_app: Page, assert_snapshot: ImageCompareFunction):
    """Test that the number cell can be edited."""
    cell_overlay_test_df = themed_app.get_by_test_id("stDataFrame").nth(3)
    expect_canvas_to_be_visible(cell_overlay_test_df)

    # Click on the first cell of the table
    click_on_cell(cell_overlay_test_df, 1, 1, double_click=True, column_width="medium")
    cell_overlay = get_open_cell_overlay(themed_app)

    # On some browsers the cell content is highlighted, so we enforce it to make the
    # test consistent and stable across all browsers
    cell_overlay.click()
    cell_overlay.press("ControlOrMeta+A")
    # Get the (number) input element and check the value
    expect(cell_overlay.locator(".gdg-input")).to_have_text("hello\nworld")
    assert_snapshot(cell_overlay, name="st_data_editor-text_col_editor")

    # Change the value
    cell_overlay.locator(".gdg-input").fill("edited value")
    # Press Enter to apply the change
    themed_app.keyboard.press("Enter")
    wait_for_app_run(themed_app)

    # Check if that the value was submitted
    expect_prefixed_markdown(
        themed_app, "Edited DF:", "edited value", exact_match=False
    )


def test_custom_css_class_via_key(app: Page):
    """Test that the element can have a custom css class via the key argument."""
    expect(get_element_by_key(app, "data_editor")).to_be_visible()


def test_column_reorder_via_ui(app: Page, assert_snapshot: ImageCompareFunction):
    """Test that columns can be reordered via drag and drop on the UI."""
    dataframe_element = app.get_by_test_id("stDataFrame").nth(0)
    expect_canvas_to_be_visible(dataframe_element)

    # 1. Move Column A behind Column C:

    # Calculate positions for source (Column A) and target (Column C) headers
    source_x, source_y = calc_middle_cell_position(0, 1, "small")  # Column A header
    target_x, target_y = calc_middle_cell_position(0, 3, "small")  # Column C header

    # Perform drag and drop using drag_to
    dataframe_element.drag_to(
        dataframe_element,
        source_position={"x": source_x, "y": source_y},
        target_position={"x": target_x, "y": target_y},
    )

    # 2. Move Column D in front of the index column:
    # This also tests that column D should get pinned since it is moved before a
    # pinned column (index column). This is visible via the grey text color.

    # Calculate positions for source (Column D) and target (Index column) headers
    source_x, source_y = calc_middle_cell_position(0, 4, "small")  # Column D header
    target_x, target_y = calc_middle_cell_position(0, 0, "small")  # Index column header

    # Perform drag and drop using drag_to
    dataframe_element.drag_to(
        dataframe_element,
        source_position={"x": source_x, "y": source_y},
        target_position={"x": target_x, "y": target_y},
    )

    # Verify column order changed by taking a screenshot
    assert_snapshot(
        dataframe_element,
        name="st_dataframe-reorder_columns_via_ui",
    )


def test_row_hover_highlight(themed_app: Page, assert_snapshot: ImageCompareFunction):
    """Test that a row gets highlighted when hovering over a cell in the row."""
    df = themed_app.get_by_test_id("stDataFrame").nth(0)
    expect_canvas_to_be_visible(df)
    column_middle_width_px, row_middle_height_px = calc_middle_cell_position(
        2, 2, "small"
    )
    df.hover(position={"x": column_middle_width_px, "y": row_middle_height_px})

    assert_snapshot(df, name="st_dataframe-row_hover_highlight")


# TODO(lukasmasuch): Add additional interactive tests:
# - Copy data to clipboard
# - Paste in data
