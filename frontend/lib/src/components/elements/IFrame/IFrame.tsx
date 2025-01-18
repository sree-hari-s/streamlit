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

import {
  isNullOrUndefined,
  notNullOrUndefined,
} from "@streamlit/lib/src/util/utils"
import { IFrame as IFrameProto } from "@streamlit/lib/src/proto"
import {
  DEFAULT_IFRAME_FEATURE_POLICY,
  DEFAULT_IFRAME_SANDBOX_POLICY,
} from "@streamlit/lib/src/util/IFrameUtil"

import { StyledIframe } from "./styled-components"

export interface IFrameProps {
  element: IFrameProto
  width: number
}

export default function IFrame({
  element,
  width: propWidth,
}: Readonly<IFrameProps>): ReactElement {
  const width = element.hasWidth ? element.width : propWidth

  // Either 'src' or 'srcDoc' will be set in our element. If 'src'
  // is set, we're loading a remote URL in the iframe.
  const src = getNonEmptyString(element.src)
  const srcDoc = notNullOrUndefined(src)
    ? undefined
    : getNonEmptyString(element.srcdoc)

  return (
    <StyledIframe
      className="stIFrame"
      data-testid="stIFrame"
      allow={DEFAULT_IFRAME_FEATURE_POLICY}
      disableScrolling={!element.scrolling}
      src={src}
      srcDoc={srcDoc}
      width={width}
      height={element.height}
      scrolling={element.scrolling ? "auto" : "no"}
      sandbox={DEFAULT_IFRAME_SANDBOX_POLICY}
      title="st.iframe"
    />
  )
}

/**
 * Return a string property from an element. If the string is
 * null or empty, return undefined instead.
 */
function getNonEmptyString(
  value: string | null | undefined
): string | undefined {
  return isNullOrUndefined(value) || value === "" ? undefined : value
}
