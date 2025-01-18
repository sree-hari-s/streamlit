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

from e2e_playwright.conftest import ImageCompareFunction, wait_for_app_loaded


def test_default_toast_rendering(
    themed_app: Page, assert_snapshot: ImageCompareFunction
):
    """Test that toasts are correctly rendered."""
    themed_app.keyboard.press("r")
    wait_for_app_loaded(themed_app)
    themed_app.wait_for_timeout(250)

    toasts = themed_app.get_by_test_id("stToast")
    expect(toasts).to_have_count(3)
    toasts.nth(2).hover()

    expect(toasts.nth(2)).to_contain_text("🐶This is a default toast message")
    assert_snapshot(toasts.nth(2), name="toast-default")


def test_collapsed_toast_rendering(
    themed_app: Page, assert_snapshot: ImageCompareFunction
):
    """Test collapsed long toasts are correctly rendered."""
    themed_app.keyboard.press("r")
    wait_for_app_loaded(themed_app)
    themed_app.wait_for_timeout(250)

    toasts = themed_app.get_by_test_id("stToast")
    expect(toasts).to_have_count(3)
    toasts.nth(1).hover()

    expect(toasts.nth(1)).to_contain_text(
        "🦄Random toast message that is a really really really really really really really long message, going wayview moreClose"
    )
    assert_snapshot(toasts.nth(1), name="toast-collapsed")


def test_expanded_toast_rendering(
    themed_app: Page, assert_snapshot: ImageCompareFunction
):
    """Test expanded long toasts are correctly rendered."""
    themed_app.keyboard.press("r")
    wait_for_app_loaded(themed_app)
    themed_app.wait_for_timeout(250)

    toasts = themed_app.get_by_test_id("stToast")
    expect(toasts).to_have_count(3)
    toasts.nth(1).hover()

    expand = themed_app.get_by_text("view more")
    expect(expand).to_have_count(1)
    expand.click()

    expect(toasts.nth(1)).to_contain_text(
        "🦄Random toast message that is a really really really really really really really long message, going way past the 3 line limitview lessClose"
    )
    assert_snapshot(toasts.nth(1), name="toast-expanded")


def test_toast_with_material_icon_rendering(
    themed_app: Page, assert_snapshot: ImageCompareFunction
):
    """Test that toasts with material icons are correctly rendered."""
    themed_app.keyboard.press("r")
    wait_for_app_loaded(themed_app)
    themed_app.wait_for_timeout(250)

    toasts = themed_app.get_by_test_id("stToast")
    expect(toasts).to_have_count(3)
    toasts.nth(0).hover()

    expect(toasts.nth(0)).to_contain_text("cabinYour edited image was saved!Close")
    assert_snapshot(toasts.nth(0), name="toast-material-icon")
