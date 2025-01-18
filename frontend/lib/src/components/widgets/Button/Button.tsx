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

import { Button as ButtonProto } from "@streamlit/lib/src/proto"
import BaseButton, {
  BaseButtonKind,
  BaseButtonSize,
  BaseButtonTooltip,
  DynamicButtonLabel,
} from "@streamlit/lib/src/components/shared/BaseButton"
import { WidgetStateManager } from "@streamlit/lib/src/WidgetStateManager"

export interface Props {
  disabled: boolean
  element: ButtonProto
  widgetMgr: WidgetStateManager
  width: number
  fragmentId?: string
}

function Button(props: Props): ReactElement {
  const { disabled, element, widgetMgr, width, fragmentId } = props
  const style = { width }

  let kind = BaseButtonKind.SECONDARY
  if (element.type === "primary") {
    kind = BaseButtonKind.PRIMARY
  } else if (element.type === "tertiary") {
    kind = BaseButtonKind.TERTIARY
  }

  // When useContainerWidth true & has help tooltip,
  // we need to pass the container width down to the button
  const fluidWidth = element.help ? width : true

  return (
    <div className="stButton" data-testid="stButton" style={style}>
      <BaseButtonTooltip help={element.help}>
        <BaseButton
          kind={kind}
          size={BaseButtonSize.SMALL}
          disabled={disabled}
          fluidWidth={element.useContainerWidth ? fluidWidth : false}
          onClick={() =>
            widgetMgr.setTriggerValue(element, { fromUi: true }, fragmentId)
          }
        >
          <DynamicButtonLabel icon={element.icon} label={element.label} />
        </BaseButton>
      </BaseButtonTooltip>
    </div>
  )
}

export default Button
