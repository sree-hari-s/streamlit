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

import React, { ReactElement } from "react"

import { screen } from "@testing-library/react"

import { render } from "@streamlit/lib/src/test_util"
import * as Utils from "@streamlit/lib/src/theme/utils"
import { mockConvertRemToPx } from "@streamlit/lib/src/mocks/mocks"

import VirtualDropdown from "./VirtualDropdown"

interface OptionProps {
  item?: { value: string }
}

function Option(props: OptionProps): ReactElement {
  return <span className={props.item ? props.item.value : "nothing"} />
}

describe("VirtualDropdown element", () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  beforeEach(() => {
    vi.spyOn(Utils, "convertRemToPx").mockImplementation(mockConvertRemToPx)
  })

  it("renders a StyledEmptyState when it has no children", () => {
    render(<VirtualDropdown />)

    expect(
      screen.getByTestId("stSelectboxVirtualDropdownEmpty")
    ).toBeInTheDocument()
  })

  it("renders a StyledEmptyState when it has children with no item", () => {
    render(
      <VirtualDropdown>
        <Option />
      </VirtualDropdown>
    )

    expect(
      screen.getByTestId("stSelectboxVirtualDropdownEmpty")
    ).toBeInTheDocument()
  })

  it("renders a FixedSizeList when it has children", () => {
    render(
      <VirtualDropdown>
        <Option item={{ value: "abc" }} />
      </VirtualDropdown>
    )

    expect(
      screen.getByTestId("stSelectboxVirtualDropdown")
    ).toBeInTheDocument()

    // each option will have a tooltip attached to it
    expect(screen.getAllByTestId("stTooltipHoverTarget")).toHaveLength(1)
  })
})
