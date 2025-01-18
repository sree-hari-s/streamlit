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
from playwright.sync_api import Page, expect

from e2e_playwright.conftest import (
    ImageCompareFunction,
    wait_for_app_loaded,
    wait_for_app_run,
)
from e2e_playwright.shared.app_utils import (
    click_button,
    click_checkbox,
    get_element_by_key,
)


def main_heading(app: Page):
    return app.get_by_test_id("stHeading").nth(0)


def page_heading(app: Page):
    return app.get_by_test_id("stHeading").nth(1)


def check_field(
    app: Page, *, hide_sidebarnav=False, dynamic_pages=False, add_sidebar_elements=False
):
    if hide_sidebarnav:
        click_checkbox(app, "Hide sidebar")

    if dynamic_pages:
        click_checkbox(app, "Change navigation dynamically")

    if add_sidebar_elements:
        click_checkbox(app, "Show sidebar elements")


expected_page_order = [
    "page 2",
    "Different Title",
    "page 4",
    "page 5",
    "slow page",
    "page 7",
    "page 8",
    "page 9",
    "page 10",
    "page 11",
    "page 12",
    "page 13",
    "page 14",
]


def get_page_link(
    app: Page, page_name: str, page_order: list[str] = expected_page_order
):
    return (
        app.get_by_test_id("stSidebarNav").locator("a").nth(page_order.index(page_name))
    )


def expect_page_order(app: Page, page_order: list[str] = expected_page_order):
    """Test that the page order is correct"""
    nav = app.get_by_test_id("stSidebarNav")
    for i, title in enumerate(page_order):
        expect(nav.locator("a").nth(i)).to_contain_text(title)


def test_loads_main_script_on_initial_page_load(app: Page):
    """Test that the main script is loaded on initial page load."""
    expect(main_heading(app)).to_contain_text("Main Page")


def test_renders_sidebar_nav_correctly(
    themed_app: Page, assert_snapshot: ImageCompareFunction
):
    """Test that the sidebar nav is rendered correctly."""
    assert_snapshot(themed_app.get_by_test_id("stSidebarNav"), name="mpa-sidebar_nav")


def test_can_switch_between_pages_by_clicking_on_sidebar_links(app: Page):
    """Test that we can switch between pages by clicking on sidebar links."""
    get_page_link(app, "Different Title").click()
    wait_for_app_run(app)
    expect(page_heading(app)).to_contain_text("Page 3")


@pytest.mark.performance
def test_switching_pages_performance(app: Page):
    """Test that we can switch between pages by clicking on sidebar links."""
    get_page_link(app, "Different Title").click()
    wait_for_app_run(app)
    expect(page_heading(app)).to_contain_text("Page 3")


def test_main_script_persists_across_page_changes(app: Page):
    """Test that we can switch between pages and content from main script persists."""
    get_page_link(app, "Different Title").click()
    wait_for_app_run(app)
    expect(main_heading(app)).to_contain_text("Main Page")


def test_main_script_widgets_persist_across_page_changes(app: Page):
    """Test that we can switch between pages and widgets from main script persists."""
    slider = app.locator('.stSlider [role="slider"]')
    slider.click()
    slider.press("ArrowRight")
    wait_for_app_run(app, wait_delay=500)

    get_page_link(app, "page 5").click()
    wait_for_app_run(app)
    expect(app.get_by_test_id("stMarkdown").nth(0)).to_contain_text("x is 1")


def test_supports_navigating_to_page_directly_via_url(app: Page, app_port: int):
    """Test that we can navigate to a page directly via URL."""
    app.goto(f"http://localhost:{app_port}/page_5")
    wait_for_app_loaded(app)

    expect(page_heading(app)).to_contain_text("Page 5")


def test_supports_navigating_to_page_directly_via_url_path(app: Page, app_port: int):
    """Test that we can navigate to a page directly via URL. using the url_path."""
    app.goto(f"http://localhost:{app_port}/my_url_path")
    wait_for_app_loaded(app)
    expect(app).to_have_url(f"http://localhost:{app_port}/my_url_path")
    expect(page_heading(app)).to_contain_text("Page 8")


def test_can_switch_between_pages_and_edit_widgets(app: Page):
    """Test that we can switch between pages and page widgets do not persist."""
    get_page_link(app, "Different Title").click()
    wait_for_app_run(app, wait_delay=1000)

    slider = app.locator('.stSlider [role="slider"]').nth(1)
    slider.click()
    slider.press("ArrowRight")
    wait_for_app_run(app)
    expect(app.get_by_test_id("stMarkdown").nth(1)).to_contain_text("x is 1")

    get_page_link(app, "page 2").click()
    wait_for_app_run(app, wait_delay=1000)

    get_page_link(app, "Different Title").click()
    wait_for_app_run(app, wait_delay=1000)

    expect(app.get_by_test_id("stMarkdown").nth(1)).to_contain_text("x is 0")


def test_titles_are_set_correctly(app: Page):
    """Test that page titles work as expected"""
    expect_page_order(app)


def test_dynamic_pages(themed_app: Page, assert_snapshot: ImageCompareFunction):
    """Test that dynamic pages are defined"""
    check_field(themed_app, dynamic_pages=True)
    wait_for_app_run(themed_app)

    nav = themed_app.get_by_test_id("stSidebarNav")
    expect_page_order(themed_app, ["page 2", "Different Title", "page 5", "page 9"])

    assert_snapshot(nav, name="dynamic-pages")


def test_show_not_found_dialog(app: Page, app_port: int):
    """Test that we show a not found dialog if the page doesn't exist."""
    app.goto(f"http://localhost:{app_port}/not_a_page")
    wait_for_app_loaded(app)

    expect(app.locator('[role="dialog"]')).to_contain_text("Page not found")


def test_handles_expand_collapse_of_mpa_nav_correctly(
    themed_app: Page, assert_snapshot: ImageCompareFunction
):
    """Test that we handle expand/collapse of MPA nav correctly."""

    check_field(themed_app, add_sidebar_elements=True)
    wait_for_app_run(themed_app)

    view_button = themed_app.get_by_test_id("stSidebarNavViewButton")

    expect(view_button).to_be_visible()

    # Expand the nav
    view_button.click(force=True)
    expect(view_button).to_have_text("View less")
    # move the mouse out of the way to avoid hover effects
    themed_app.mouse.move(0, 0)
    assert_snapshot(
        themed_app.get_by_test_id("stSidebarNav"), name="mpa-sidebar_nav_expanded"
    )

    # Collapse the nav
    view_button.click(force=True)
    expect(view_button).to_have_text("View 3 more")
    # move the mouse out of the way to avoid hover effects
    themed_app.mouse.move(0, 0)
    assert_snapshot(
        themed_app.get_by_test_id("stSidebarNav"), name="mpa-sidebar_nav_collapsed"
    )

    # Expand the nav again
    view_button.click(force=True)
    expect(view_button).to_have_text("View less")
    # move the mouse out of the way to avoid hover effects
    themed_app.mouse.move(0, 0)
    assert_snapshot(
        themed_app.get_by_test_id("stSidebarNav"), name="mpa-sidebar_nav_expanded"
    )


def test_handles_expanded_navigation_parameter_correctly(app: Page):
    """Test that we handle expanded param of st.navigation nav correctly."""

    click_checkbox(app, "Show sidebar elements")
    wait_for_app_run(app)

    # By default, the navigation is collapsed
    view_button = app.get_by_test_id("stSidebarNavViewButton")
    expect(view_button).to_be_visible()

    links = app.get_by_test_id("stSidebarNav").locator("a")
    expect(links).to_have_count(10)

    # Forced expansion removes the View less button and shows all links
    click_checkbox(app, "Expand navigation")
    wait_for_app_run(app)

    view_button = app.get_by_test_id("stSidebarNavViewButton")

    expect(view_button).not_to_be_visible()
    links = app.get_by_test_id("stSidebarNav").locator("a")
    expect(links).to_have_count(13)

    # Removing forced expansion shows the View less button but remains expanded
    click_checkbox(app, "Expand navigation")
    wait_for_app_run(app)
    view_button = app.get_by_test_id("stSidebarNavViewButton")

    expect(view_button).to_be_visible()
    links = app.get_by_test_id("stSidebarNav").locator("a")
    expect(links).to_have_count(13)


def test_preserves_navigation_expansion_user_preference(app: Page, app_port: int):
    """Test that the navigation expansion state is preserved across page changes."""
    click_checkbox(app, "Show sidebar elements")
    wait_for_app_run(app)

    # verify the default setting is collapsed
    view_more_button = app.get_by_test_id("stSidebarNavViewButton")
    expect(view_more_button).to_be_visible()
    links = app.get_by_test_id("stSidebarNav").locator("a")
    expect(links).to_have_count(10)

    # User clicks View more which preserves the setting
    view_more_button.click()

    # Verify navigation is expanded
    view_less_button = app.get_by_test_id("stSidebarNavViewButton")
    expect(view_less_button).to_have_text("View less")
    links = app.get_by_test_id("stSidebarNav").locator("a")
    expect(links).to_have_count(13)

    # Reload the page and ensure elements are in the sidebar
    app.goto(f"http://localhost:{app_port}")
    wait_for_app_loaded(app)

    click_checkbox(app, "Show sidebar elements")
    wait_for_app_run(app)

    # Verify navigation remains expanded
    links = app.get_by_test_id("stSidebarNav").locator("a")
    expect(links).to_have_count(13)
    view_less_button = app.get_by_test_id("stSidebarNavViewButton")
    expect(view_less_button).to_have_text("View less")

    # Undo the setting (eliminating the preference)
    view_less_button.click()

    # Verify navigation is collapsed
    view_less_button = app.get_by_test_id("stSidebarNavViewButton")
    expect(view_less_button).to_have_text("View 3 more")
    links = app.get_by_test_id("stSidebarNav").locator("a")
    expect(links).to_have_count(10)

    # Reload the page and ensure elements are in the sidebar
    app.goto(f"http://localhost:{app_port}")
    wait_for_app_loaded(app)

    click_checkbox(app, "Show sidebar elements")
    wait_for_app_run(app)

    links = app.get_by_test_id("stSidebarNav").locator("a")
    expect(links).to_have_count(10)
    expect(app.get_by_test_id("stSidebarNavViewButton")).to_have_text("View 3 more")


def test_switch_page_by_path(app: Page):
    """Test that we can switch between pages by triggering st.switch_page with a path."""

    click_button(app, "page 5")

    expect(page_heading(app)).to_contain_text("Page 5")


def test_switch_page_by_st_page(app: Page):
    """Test that we can switch between pages by triggering st.switch_page with st.Page."""

    click_button(app, "page 9")

    expect(page_heading(app)).to_contain_text("Page 9")


def test_removes_query_params_with_st_switch_page(app: Page, app_port: int):
    """Test that query params are removed when navigating via st.switch_page"""

    # Start at main page with query params
    app.goto(f"http://localhost:{app_port}/?foo=bar")
    wait_for_app_loaded(app)
    expect(app).to_have_url(f"http://localhost:{app_port}/?foo=bar")

    # Trigger st.switch_page
    click_button(app, "page 5")

    # Check that query params don't persist
    expect(app).to_have_url(f"http://localhost:{app_port}/page_5")


def test_removes_query_params_when_clicking_link(app: Page, app_port: int):
    """Test that query params are removed when swapping pages by clicking on a link"""

    app.goto(f"http://localhost:{app_port}/page_7?foo=bar")
    wait_for_app_loaded(app)
    expect(app).to_have_url(f"http://localhost:{app_port}/page_7?foo=bar")

    get_page_link(app, "page 4").click()
    wait_for_app_loaded(app)
    expect(app).to_have_url(f"http://localhost:{app_port}/page_4")


def test_removes_non_embed_query_params_when_swapping_pages(app: Page, app_port: int):
    """Test that non-embed query params are removed when swapping pages"""

    app.goto(
        f"http://localhost:{app_port}/page_7?foo=bar&embed=True&embed_options=show_toolbar&embed_options=show_colored_line"
    )
    wait_for_app_loaded(app)
    expect(app).to_have_url(
        f"http://localhost:{app_port}/page_7?foo=bar&embed=True&embed_options=show_toolbar&embed_options=show_colored_line"
    )

    get_page_link(app, "page 4").click()
    wait_for_app_loaded(app)

    expect(app).to_have_url(
        f"http://localhost:{app_port}/page_4?embed=true&embed_options=show_toolbar&embed_options=show_colored_line"
    )


def test_renders_logos(app: Page, assert_snapshot: ImageCompareFunction):
    """Test that logos display properly in sidebar and main sections"""

    # Go to logo page & wait short moment for logo to appear
    get_page_link(app, "page 8").click()
    wait_for_app_loaded(app)

    # Sidebar logo
    expect(app.get_by_test_id("stSidebarHeader").locator("a")).to_have_attribute(
        "href", "https://www.example.com"
    )
    assert_snapshot(app.get_by_test_id("stSidebar"), name="sidebar-logo")

    # Collapse the sidebar
    app.get_by_test_id("stSidebarContent").hover()
    app.get_by_test_id("stSidebarCollapseButton").locator("button").click()
    app.wait_for_timeout(500)

    # Collapsed logo
    expect(
        app.get_by_test_id("stSidebarCollapsedControl").locator("a")
    ).to_have_attribute("href", "https://www.example.com")
    assert_snapshot(
        app.get_by_test_id("stSidebarCollapsedControl"), name="collapsed-logo"
    )


def test_page_link_with_path(app: Page):
    """Test st.page_link works with a path"""

    app.get_by_test_id("stPageLink-NavLink").filter(has_text="page 5 page link").click()
    wait_for_app_loaded(app)

    expect(page_heading(app)).to_contain_text("Page 5")


def test_page_link_with_st_file(app: Page):
    """Test st.page_link works with a st.Page"""

    app.get_by_test_id("stPageLink-NavLink").filter(has_text="page 9 page link").click()
    wait_for_app_loaded(app)

    expect(page_heading(app)).to_contain_text("Page 9")


def test_hidden_navigation(app: Page):
    """Test position=hidden hides the navigation"""

    expect(app.get_by_test_id("stSidebarNav")).to_be_visible()
    check_field(app, add_sidebar_elements=True)
    wait_for_app_run(app)

    check_field(app, hide_sidebarnav=True)
    wait_for_app_run(app)

    nav_exists = app.get_by_test_id("stSidebarNav")
    expect(nav_exists).not_to_be_attached()


def test_set_default_navigation(app: Page, app_port: int):
    """Test the default page set will be shown on initial load"""

    expect(page_heading(app)).to_contain_text("Page 2")
    wait_for_app_run(app)

    app.goto(f"http://localhost:{app_port}/?default=True")
    wait_for_app_loaded(app)

    expect(page_heading(app)).to_contain_text("Page 7")


def test_page_url_path_appears_in_url(app: Page, app_port: int):
    """Test that st.Page's url_path is included in the URL"""
    link = get_page_link(app, "page 8")

    expect(link).to_have_attribute("href", f"http://localhost:{app_port}/my_url_path")
    link.click()
    wait_for_app_loaded(app)
    expect(app).to_have_url(f"http://localhost:{app_port}/my_url_path")


def test_widgets_maintain_state_in_fragment(app: Page):
    """Test that widgets maintain state in a fragment"""
    get_page_link(app, "page 10").click()

    input = app.get_by_test_id("stTextInput").locator("input").first
    input.fill("Hello")
    input.blur()
    wait_for_app_run(app)

    expect(input).to_have_value("Hello")


def test_widget_state_reset_on_page_switch(app: Page):
    # Regression test for GH issue 7338 for MPAv2

    slider = app.locator('.stSlider [role="slider"]')
    slider.click()
    slider.press("ArrowRight")
    wait_for_app_run(app, wait_delay=500)
    expect(app.get_by_text("x is 1")).to_be_attached()

    get_page_link(app, "slow page").click()

    # Wait for the view container and main menu to appear (like in wait_for_app_loaded),
    # but don't wait for the script to finish running.
    app.wait_for_selector(
        "[data-testid='stAppViewContainer']", timeout=30000, state="attached"
    )
    app.wait_for_selector("[data-testid='stMainMenu']", timeout=20000, state="attached")

    # Back to page 2
    get_page_link(app, "page 2").click()
    wait_for_app_run(app, wait_delay=500)

    # Slider reset
    expect(app.get_by_text("x is 1")).to_be_attached()


def test_rapid_fire_interaction(app: Page):
    """Check that the number input can handle rapid fire clicks in an Multipage app."""
    get_page_link(app, "page 14").click()

    number_input = get_element_by_key(app, "mynum")
    step_up_btn = number_input.get_by_test_id("stNumberInputStepUp")

    # we need to have the clicking last a long enough time
    for _ in range(30):
        step_up_btn.click()

    wait_for_app_run(app)

    expect(number_input.locator("input")).to_have_value("31")


def test_rapid_fire_interaction_in_fragment(app: Page):
    """Check that the number input in a fragment can handle rapid fire clicks in an Multipage app."""
    get_page_link(app, "page 14").click()

    number_input = get_element_by_key(app, "mynum2")
    step_up_btn = number_input.get_by_test_id("stNumberInputStepUp")

    # we need to have the clicking last a long enough time
    for _ in range(30):
        step_up_btn.click()

    wait_for_app_run(app)

    expect(number_input.locator("input")).to_have_value("31")


@pytest.mark.performance
def test_sidebar_interaction_performance(app: Page):
    """
    Test the performance of the sidebar interaction.
    As of writing, there is a re-rendering issue in the Sidebar where every
    option is re-rendered in the sidebar when something is hovered. This
    performance test gives us a way to measure performance improvements.
    """
    sidebar = app.get_by_test_id("stSidebar")
    options = sidebar.locator("li")
    for option in options.all():
        option.hover()
