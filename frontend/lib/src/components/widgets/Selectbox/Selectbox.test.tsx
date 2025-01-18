/**
 * Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022-2025)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import React from "react"

import { act, fireEvent, screen } from "@testing-library/react"
import { userEvent } from "@testing-library/user-event"

import { render } from "@streamlit/lib/src/test_util"
import { WidgetStateManager } from "@streamlit/lib/src/WidgetStateManager"
import { Selectbox as SelectboxProto } from "@streamlit/lib/src/proto"
import * as Utils from "@streamlit/lib/src/theme/utils"
import { mockConvertRemToPx } from "@streamlit/lib/src/mocks/mocks"

import Selectbox, { Props } from "./Selectbox"

const getProps = (
  elementProps: Partial<SelectboxProto> = {},
  widgetProps: Partial<Props> = {}
): Props => ({
  element: SelectboxProto.create({
    id: "1",
    label: "Label",
    default: 0,
    options: ["a", "b", "c"],
    ...elementProps,
  }),
  width: 0,
  disabled: false,
  widgetMgr: new WidgetStateManager({
    sendRerunBackMsg: vi.fn(),
    formsDataChanged: vi.fn(),
  }),
  ...widgetProps,
})

const pickOption = (selectbox: HTMLElement, value: string): void => {
  // TODO: Utilize userEvent instead of fireEvent. This somehow fails with userEvent.
  // eslint-disable-next-line testing-library/prefer-user-event
  fireEvent.click(selectbox)
  const valueElement = screen.getByText(value)
  // TODO: Utilize userEvent instead of fireEvent. This somehow fails with userEvent.
  // eslint-disable-next-line testing-library/prefer-user-event
  fireEvent.click(valueElement)
}

describe("Selectbox widget", () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("renders without crashing", () => {
    const props = getProps()
    render(<Selectbox {...props} />)
    const selectbox = screen.getByTestId("stSelectbox")
    expect(selectbox).toBeInTheDocument()
    expect(selectbox).toHaveClass("stSelectbox")
  })

  it("sets widget value on mount", () => {
    const props = getProps()
    vi.spyOn(props.widgetMgr, "setIntValue")

    render(<Selectbox {...props} />)
    expect(props.widgetMgr.setIntValue).toHaveBeenCalledWith(
      props.element,
      props.element.default,
      { fromUi: false },
      undefined
    )
  })

  it("can pass fragmentId to setIntValue", () => {
    const props = getProps(undefined, { fragmentId: "myFragmentId" })
    vi.spyOn(props.widgetMgr, "setIntValue")

    render(<Selectbox {...props} />)
    expect(props.widgetMgr.setIntValue).toHaveBeenCalledWith(
      props.element,
      props.element.default,
      { fromUi: false },
      "myFragmentId"
    )
  })

  it("handles the onChange event", () => {
    const props = getProps()
    vi.spyOn(props.widgetMgr, "setIntValue")
    vi.spyOn(Utils, "convertRemToPx").mockImplementation(mockConvertRemToPx)

    render(<Selectbox {...props} />)

    const selectbox = screen.getByRole("combobox")

    pickOption(selectbox, "b")

    expect(props.widgetMgr.setIntValue).toHaveBeenLastCalledWith(
      props.element,
      1,
      { fromUi: true },
      undefined
    )
    expect(screen.queryByText("a")).not.toBeInTheDocument()
    expect(screen.getByText("b")).toBeInTheDocument()
  })

  it("resets its value when form is cleared", () => {
    // Create a widget in a clearOnSubmit form
    const props = getProps({ formId: "form" })
    props.widgetMgr.setFormSubmitBehaviors("form", true)

    vi.spyOn(props.widgetMgr, "setIntValue")
    vi.spyOn(Utils, "convertRemToPx").mockImplementation(mockConvertRemToPx)

    render(<Selectbox {...props} />)

    const selectbox = screen.getByRole("combobox")
    pickOption(selectbox, "b")

    expect(props.widgetMgr.setIntValue).toHaveBeenLastCalledWith(
      props.element,
      1,
      { fromUi: true },
      undefined
    )

    // "Submit" the form
    act(() => {
      props.widgetMgr.submitForm("form", undefined)
    })

    // Our widget should be reset, and the widgetMgr should be updated
    expect(screen.getByText("a")).toBeInTheDocument()
    expect(screen.queryByText("b")).not.toBeInTheDocument()
    expect(props.widgetMgr.setIntValue).toHaveBeenLastCalledWith(
      props.element,
      props.element.default,
      {
        fromUi: true,
      },
      undefined
    )
  })

  it("maintains scroll position when reopening dropdown", async () => {
    const user = userEvent.setup()
    const props = getProps({
      options: Array.from({ length: 100 }, (_, i) => `Option ${i}`),
    })
    vi.spyOn(Utils, "convertRemToPx").mockImplementation(mockConvertRemToPx)

    render(<Selectbox {...props} />)
    const selectbox = screen.getByRole("combobox")

    // Open dropdown
    await user.click(selectbox)

    // Get dropdown content and scroll
    const dropdown = screen.getByTestId("stSelectboxVirtualDropdown")
    act(() => {
      // Simulate scrolling down
      const scrollEvent = new Event("scroll", { bubbles: true })
      Object.defineProperty(dropdown, "scrollTop", { value: 500 })
      dropdown.dispatchEvent(scrollEvent)
    })

    // Close dropdown
    await user.keyboard("{Escape}")

    // Reopen dropdown
    await user.click(selectbox)

    // Check if scroll position was maintained
    expect(dropdown.scrollTop).toBe(500)
  })
})
