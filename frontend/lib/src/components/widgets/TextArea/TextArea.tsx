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

import React, { FC, memo, useCallback, useRef, useState } from "react"

import { Textarea as UITextArea } from "baseui/textarea"
import { useTheme } from "@emotion/react"
import uniqueId from "lodash/uniqueId"

import { TextArea as TextAreaProto } from "@streamlit/lib/src/proto"
import { WidgetStateManager } from "@streamlit/lib/src/WidgetStateManager"
import useUpdateUiValue from "@streamlit/lib/src/hooks/useUpdateUiValue"
import useSubmitFormViaEnterKey from "@streamlit/lib/src/hooks/useSubmitFormViaEnterKey"
import useOnInputChange from "@streamlit/lib/src/hooks/useOnInputChange"
import InputInstructions from "@streamlit/lib/src/components/shared/InputInstructions/InputInstructions"
import {
  StyledWidgetLabelHelp,
  WidgetLabel,
} from "@streamlit/lib/src/components/widgets/BaseWidget"
import TooltipIcon from "@streamlit/lib/src/components/shared/TooltipIcon"
import { Placement } from "@streamlit/lib/src/components/shared/Tooltip"
import {
  isInForm,
  labelVisibilityProtoValueToEnum,
} from "@streamlit/lib/src/util/utils"
import { EmotionTheme } from "@streamlit/lib/src/theme"
import {
  useBasicWidgetState,
  ValueWithSource,
} from "@streamlit/lib/src/hooks/useBasicWidgetState"

export interface Props {
  disabled: boolean
  element: TextAreaProto
  widgetMgr: WidgetStateManager
  width: number
  fragmentId?: string
}

type TextAreaValue = string | null

const getStateFromWidgetMgr = (
  widgetMgr: WidgetStateManager,
  element: TextAreaProto
): TextAreaValue | undefined => {
  return widgetMgr.getStringValue(element) ?? element.default ?? null
}

const getDefaultStateFromProto = (element: TextAreaProto): TextAreaValue => {
  return element.default ?? null
}

const getCurrStateFromProto = (element: TextAreaProto): TextAreaValue => {
  return element.value ?? null
}

const updateWidgetMgrState = (
  element: TextAreaProto,
  widgetMgr: WidgetStateManager,
  valueWithSource: ValueWithSource<TextAreaValue>,
  fragmentId?: string
): void => {
  widgetMgr.setStringValue(
    element,
    valueWithSource.value,
    { fromUi: valueWithSource.fromUi },
    fragmentId
  )
}

const TextArea: FC<Props> = ({
  disabled,
  element,
  widgetMgr,
  fragmentId,
  width,
}) => {
  // TODO: Update to match React best practices
  // eslint-disable-next-line react-compiler/react-compiler
  const id = useRef(uniqueId("text_area_")).current

  /**
   * True if the user-specified state.value has not yet been synced to the WidgetStateManager.
   */
  const [dirty, setDirty] = useState(false)
  /**
   * Whether the area is currently focused.
   */
  const [focused, setFocused] = useState(false)

  /**
   * The value specified by the user via the UI. If the user didn't touch this
   * widget's UI, the default value is used.
   */
  const [uiValue, setUiValue] = useState<string | null>(
    getStateFromWidgetMgr(widgetMgr, element) ?? null
  )

  const onFormCleared = useCallback(() => {
    setUiValue(element.default ?? null)
    setDirty(true)
  }, [element])

  const [value, setValueWithSource] = useBasicWidgetState<
    TextAreaValue,
    TextAreaProto
  >({
    getStateFromWidgetMgr,
    getDefaultStateFromProto,
    getCurrStateFromProto,
    updateWidgetMgrState,
    element,
    widgetMgr,
    fragmentId,
    onFormCleared,
  })

  useUpdateUiValue(value, uiValue, setUiValue, dirty)

  const theme: EmotionTheme = useTheme()

  const commitWidgetValue = useCallback((): void => {
    setDirty(false)
    setValueWithSource({ value: uiValue, fromUi: true })
  }, [uiValue, setValueWithSource])

  const onBlur = useCallback(() => {
    if (dirty) {
      commitWidgetValue()
    }
    setFocused(false)
  }, [dirty, commitWidgetValue])

  const onFocus = useCallback(() => {
    setFocused(true)
  }, [])

  const onChange = useOnInputChange({
    formId: element.formId,
    maxChars: element.maxChars,
    setDirty,
    setUiValue,
    setValueWithSource,
  })

  const onKeyDown = useSubmitFormViaEnterKey(
    element.formId,
    commitWidgetValue,
    dirty,
    widgetMgr,
    fragmentId,
    true
  )

  const style = { width }
  const { height, placeholder, formId } = element

  // Show "Please enter" instructions if in a form & allowed, or not in form and state is dirty.
  const allowEnterToSubmit = isInForm({ formId })
    ? widgetMgr.allowFormEnterToSubmit(formId)
    : dirty

  // Hide input instructions for small widget sizes.
  const shouldShowInstructions =
    focused && width > theme.breakpoints.hideWidgetDetails

  return (
    <div className="stTextArea" data-testid="stTextArea" style={style}>
      <WidgetLabel
        label={element.label}
        disabled={disabled}
        labelVisibility={labelVisibilityProtoValueToEnum(
          element.labelVisibility?.value
        )}
        htmlFor={id}
      >
        {element.help && (
          <StyledWidgetLabelHelp>
            <TooltipIcon
              content={element.help}
              placement={Placement.TOP_RIGHT}
            />
          </StyledWidgetLabelHelp>
        )}
      </WidgetLabel>
      <UITextArea
        value={uiValue ?? ""}
        placeholder={placeholder}
        onBlur={onBlur}
        onFocus={onFocus}
        onChange={onChange}
        onKeyDown={onKeyDown}
        aria-label={element.label}
        disabled={disabled}
        id={id}
        overrides={{
          Input: {
            style: {
              lineHeight: theme.lineHeights.inputWidget,

              // The default height of the text area is calculated to perfectly fit 3 lines of text.
              height: height ? `${height}px` : "",
              minHeight: theme.sizes.largestElementHeight,
              resize: "vertical",
              "::placeholder": {
                opacity: "0.7",
              },
              // Baseweb requires long-hand props, short-hand leads to weird bugs & warnings.
              paddingRight: theme.spacing.lg,
              paddingLeft: theme.spacing.lg,
              paddingBottom: theme.spacing.lg,
              paddingTop: theme.spacing.lg,
            },
          },
          Root: {
            props: {
              "data-testid": "stTextAreaRootElement",
            },
            style: {
              // Baseweb requires long-hand props, short-hand leads to weird bugs & warnings.
              borderLeftWidth: theme.sizes.borderWidth,
              borderRightWidth: theme.sizes.borderWidth,
              borderTopWidth: theme.sizes.borderWidth,
              borderBottomWidth: theme.sizes.borderWidth,
            },
          },
        }}
      />
      {shouldShowInstructions && (
        <InputInstructions
          dirty={dirty}
          value={uiValue ?? ""}
          maxLength={element.maxChars}
          type={"multiline"}
          inForm={isInForm({ formId })}
          allowEnterToSubmit={allowEnterToSubmit}
        />
      )}
    </div>
  )
}

export default memo(TextArea)
