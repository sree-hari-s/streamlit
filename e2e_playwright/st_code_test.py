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


def test_code_display(app: Page):
    """Test that st.code displays a code block."""
    code_element = app.get_by_test_id("stCode").first
    expect(code_element).to_contain_text("This code is awesome!")


def test_syntax_highlighting(themed_app: Page, assert_snapshot: ImageCompareFunction):
    """Test that the copy-to-clipboard action appears on hover."""
    first_code_element = themed_app.get_by_test_id("stCode").first
    first_code_element.hover()
    assert_snapshot(first_code_element, name="st_code-hover_copy")


def test_code_blocks_render_correctly(
    themed_app: Page, assert_snapshot: ImageCompareFunction
):
    """Test that the code blocks render as expected via screenshot matching."""
    code_blocks = themed_app.get_by_test_id("stCode")
    expect(code_blocks).to_have_count(17)
    # The code blocks might require a bit more time for rendering, so wait until
    # the text is truly visible. Otherwise we might get blank code blocks in the
    # screenshots.
    foo_func_count = 5
    themed_app.wait_for_function(
        f"()=>document.body.textContent.split('def foo()').length === {foo_func_count}"
    )

    assert_snapshot(code_blocks.nth(0), name="st_code-auto_lang")
    assert_snapshot(code_blocks.nth(1), name="st_code-empty")
    assert_snapshot(code_blocks.nth(2), name="st_code-python_lang")
    assert_snapshot(code_blocks.nth(3), name="st_code-line_numbers")
    assert_snapshot(code_blocks.nth(4), name="st_code-no_lang")
    assert_snapshot(code_blocks.nth(5), name="st_markdown-code_block")
    assert_snapshot(code_blocks.nth(6), name="st_code-diff_lang")

    # Test long lines draw as expected.
    assert_snapshot(code_blocks.nth(11), name="st_code-long-no_wrap")
    assert_snapshot(code_blocks.nth(12), name="st_code-long-numbers-no_wrap")
    assert_snapshot(code_blocks.nth(13), name="st_code-long-wrap")
    assert_snapshot(code_blocks.nth(14), name="st_code-long-numbers-wrap")

    # Test height prop
    assert_snapshot(code_blocks.nth(15), name="st_code-height-long-code")
    assert_snapshot(code_blocks.nth(16), name="st_code-height-short-code")


def test_correct_bottom_spacing_for_code_blocks(app: Page):
    """Test that the code blocks have the correct bottom spacing."""

    # The first code block should have no bottom margin:
    expect(
        app.get_by_test_id("stExpander").nth(0).get_by_test_id("stCode").first
    ).to_have_css("margin-bottom", "0px")
    # While the codeblock used inside markdown should have a bottom margin to imitate the gap:
    expect(
        app.get_by_test_id("stExpander").nth(1).get_by_test_id("stMarkdownPre").first
    ).to_have_css("margin-bottom", "16px")


def test_check_top_level_class(app: Page):
    """Check that the top level class is correctly set."""
    check_top_level_class(app, "stCode")


def test_line_wrap(app: Page):
    """Test that line-wrapping works correctly."""

    code_blocks = app.get_by_test_id("stCode")

    # When line-wrap is off, the "EOL" token should not be visible.

    curr_block = code_blocks.nth(11)
    curr_block.scroll_into_view_if_needed()
    expect(curr_block.get_by_text("EOL")).not_to_be_in_viewport()

    curr_block = code_blocks.nth(12)
    curr_block.scroll_into_view_if_needed()
    expect(curr_block.get_by_text("EOL")).not_to_be_in_viewport()

    # When line-wrap is on, the "EOL" token should be visible.

    curr_block = code_blocks.nth(13)
    curr_block.scroll_into_view_if_needed()
    expect(curr_block.get_by_text("EOL")).to_be_in_viewport()

    curr_block = code_blocks.nth(14)
    curr_block.scroll_into_view_if_needed()
    expect(curr_block.get_by_text("EOL")).to_be_in_viewport()


def test_height_parameter(app: Page):
    """Test that the height prop works correctly."""
    code_blocks = app.get_by_test_id("stCode")

    # Test long code with fixed height
    curr_block = code_blocks.nth(15)
    curr_block.scroll_into_view_if_needed()
    expect(curr_block.locator("pre")).to_have_css("height", "200px")
    # The "EOL" token at the end of the code block should not be visible.
    expect(curr_block.get_by_text("EOL")).not_to_be_in_viewport()

    # Test short code with fixed height
    curr_block = code_blocks.nth(16)
    curr_block.scroll_into_view_if_needed()
    expect(curr_block.locator("pre")).to_have_css("height", "200px")
