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

from e2e_playwright.shared.app_utils import click_button, click_checkbox, get_checkbox


def test_checking_checkbox_unchecks_other(app: Page):
    """Test that checking one checkbox unchecks the other by using callbacks."""
    first_checkbox = get_checkbox(app, "Checkbox1")
    second_checkbox = get_checkbox(app, "Checkbox2")

    expect(first_checkbox.locator("input")).to_have_attribute("aria-checked", "true")
    expect(second_checkbox.locator("input")).to_have_attribute("aria-checked", "false")

    click_checkbox(app, "Checkbox2")

    expect(first_checkbox.locator("input")).to_have_attribute("aria-checked", "false")
    expect(second_checkbox.locator("input")).to_have_attribute("aria-checked", "true")


def test_has_correct_starting_values(app: Page):
    expect(app.get_by_text("item_counter: 0")).to_have_count(1)
    expect(app.get_by_text("attr_counter: 0")).to_have_count(1)
    expect(app.get_by_text("len(st.session_state): 5")).to_have_count(1)
    expect(app.get_by_test_id("stJson")).to_be_visible()


def test_can_do_CRUD_for_session_state_items(app: Page):
    expect(app.get_by_text("item_counter: 0")).to_have_count(1)
    expect(app.get_by_text("attr_counter: 0")).to_have_count(1)

    click_button(app, "inc_item_counter")

    expect(app.get_by_text("item_counter: 1")).to_have_count(1)
    expect(app.get_by_text("attr_counter: 0")).to_have_count(1)

    click_button(app, "inc_item_counter")

    expect(app.get_by_text("item_counter: 2")).to_have_count(1)
    expect(app.get_by_text("attr_counter: 0")).to_have_count(1)

    click_button(app, "del_item_counter")

    expect(app.get_by_text("item_counter: 2")).to_have_count(0)
    expect(app.get_by_text("attr_counter: 0")).to_have_count(1)
    expect(app.get_by_text("len(st.session_state): 4")).to_have_count(1)


def test_can_do_CRUD_for_session_state_attributes(app: Page):
    expect(app.get_by_text("item_counter: 0")).to_have_count(1)
    expect(app.get_by_text("attr_counter: 0")).to_have_count(1)

    click_button(app, "inc_attr_counter")

    expect(app.get_by_text("item_counter: 0")).to_have_count(1)
    expect(app.get_by_text("attr_counter: 1")).to_have_count(1)

    click_button(app, "inc_attr_counter")

    expect(app.get_by_text("item_counter: 0")).to_have_count(1)
    expect(app.get_by_text("attr_counter: 2")).to_have_count(1)

    click_button(app, "del_attr_counter")

    expect(app.get_by_text("item_counter: 0")).to_have_count(1)
    expect(app.get_by_text("attr_counter: 2")).to_have_count(0)
    expect(app.get_by_text("len(st.session_state): 4")).to_have_count(1)
