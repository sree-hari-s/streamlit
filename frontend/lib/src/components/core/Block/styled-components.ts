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

import styled from "@emotion/styled"

import { StyledCheckbox } from "@streamlit/lib/src/components/widgets/Checkbox/styled-components"
import { Block as BlockProto } from "@streamlit/lib/src/proto"
import { EmotionTheme, STALE_STYLES } from "@streamlit/lib/src/theme"

function translateGapWidth(gap: string, theme: EmotionTheme): string {
  let gapWidth = theme.spacing.lg
  if (gap === "medium") {
    gapWidth = theme.spacing.threeXL
  } else if (gap === "large") {
    gapWidth = theme.spacing.fourXL
  }
  return gapWidth
}
export interface StyledHorizontalBlockProps {
  gap: string
}

export const StyledHorizontalBlock = styled.div<StyledHorizontalBlockProps>(
  ({ theme, gap }) => {
    const gapWidth = translateGapWidth(gap, theme)

    return {
      // While using flex for columns, padding is used for large screens and gap
      // for small ones. This can be adjusted once more information is passed.
      // More information and discussions can be found: Issue #2716, PR #2811
      display: "flex",
      flexWrap: "wrap",
      flexGrow: 1,
      alignItems: "stretch",
      gap: gapWidth,
    }
  }
)

export interface StyledElementContainerProps {
  isStale: boolean
  width: number
  elementType: string
}

const GLOBAL_ELEMENTS = ["balloons", "snow"]
export const StyledElementContainer = styled.div<StyledElementContainerProps>(
  ({ theme, isStale, width, elementType }) => ({
    width,
    // Allows to have absolutely-positioned nodes inside app elements, like
    // floating buttons.
    position: "relative",

    "@media print": {
      overflow: "visible",
    },

    ":is(.stHtml-empty)": {
      display: "none",
    },

    ":has(> .stCacheSpinner)": {
      height: theme.spacing.none,
      overflow: "visible",
      visibility: "visible",
      marginBottom: `-${theme.spacing.lg}`,
      zIndex: theme.zIndices.cacheSpinner,
    },

    ":has(> .stPageLink)": {
      marginTop: `-${theme.spacing.xs}`,
      marginBottom: `-${theme.spacing.xs}`,
    },

    ...(isStale && elementType !== "skeleton" && STALE_STYLES),
    ...(elementType === "empty"
      ? {
          // Use display: none for empty elements to avoid the flexbox gap.
          display: "none",
        }
      : {}),
    ...(GLOBAL_ELEMENTS.includes(elementType)
      ? {
          // Global elements are rendered in their delta position, but they
          // are not part of the flexbox layout. We apply a negative margin
          // to remove the flexbox gap. display: none does not work for these,
          // since they needs to be visible.
          marginBottom: `-${theme.spacing.lg}`,
        }
      : {}),
  })
)

interface StyledColumnProps {
  weight: number
  gap: string
  showBorder: boolean
  verticalAlignment?: BlockProto.Column.VerticalAlignment
}

export const StyledColumn = styled.div<StyledColumnProps>(
  ({ theme, weight, gap, showBorder, verticalAlignment }) => {
    const { VerticalAlignment } = BlockProto.Column
    const percentage = weight * 100
    const gapWidth = translateGapWidth(gap, theme)
    const width = `calc(${percentage}% - ${gapWidth})`

    return {
      // Calculate width based on percentage, but fill all available space,
      // e.g. if it overflows to next row.
      width,
      flex: `1 1 ${width}`,

      [`@media (max-width: ${theme.breakpoints.columns})`]: {
        minWidth: `calc(100% - ${theme.spacing.twoXL})`,
      },
      ...(verticalAlignment === VerticalAlignment.BOTTOM && {
        marginTop: "auto",
        // Add margin to the first checkbox/toggle within the column to align it
        // better with other input widgets.
        [`& ${StyledElementContainer}:last-of-type > ${StyledCheckbox}`]: {
          marginBottom: theme.spacing.sm,
        },
      }),
      ...(verticalAlignment === VerticalAlignment.TOP && {
        // Add margin to the first checkbox/toggle within the column to align it
        // better with other input widgets.
        [`& ${StyledElementContainer}:last-of-type > ${StyledCheckbox}`]: {
          marginTop: theme.spacing.sm,
        },
      }),
      ...(verticalAlignment === VerticalAlignment.CENTER && {
        marginTop: "auto",
        marginBottom: "auto",
      }),
      ...(showBorder && {
        border: `${theme.sizes.borderWidth} solid ${theme.colors.borderColor}`,
        borderRadius: theme.radii.default,
        padding: `calc(${theme.spacing.lg} - ${theme.sizes.borderWidth})`,
      }),
    }
  }
)

export interface StyledVerticalBlockProps {
  ref?: React.RefObject<any>
  width?: number
}

export const StyledVerticalBlock = styled.div<StyledVerticalBlockProps>(
  ({ width, theme }) => ({
    width,
    position: "relative", // Required for the automatic width computation.
    display: "flex",
    flex: 1,
    flexDirection: "column",
    gap: theme.spacing.lg,
  })
)

export const StyledVerticalBlockWrapper = styled.div<StyledVerticalBlockProps>(
  {
    display: "flex",
    flexDirection: "column",
    flex: 1,
  }
)

export interface StyledVerticalBlockBorderWrapperProps {
  border: boolean
  height?: number
}

export const StyledVerticalBlockBorderWrapper =
  styled.div<StyledVerticalBlockBorderWrapperProps>(
    ({ theme, border, height }) => ({
      ...(border && {
        border: `${theme.sizes.borderWidth} solid ${theme.colors.borderColor}`,
        borderRadius: theme.radii.default,
        padding: `calc(${theme.spacing.lg} - ${theme.sizes.borderWidth})`,
      }),
      ...(height && {
        height: `${height}px`,
        overflow: "auto",
      }),
    })
  )
