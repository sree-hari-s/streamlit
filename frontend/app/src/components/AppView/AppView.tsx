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
  AppRoot,
  BlockNode,
  ComponentRegistry,
  FileUploadClient,
  FormsData,
  IAppPage,
  IGuestToHostMessage,
  LibContext,
  Logo,
  Profiler,
  ScriptRunState,
  StreamlitEndpoints,
  VerticalBlock,
  WidgetStateManager,
} from "@streamlit/lib"
import { ThemedSidebar } from "@streamlit/app/src/components/Sidebar"
import EventContainer from "@streamlit/app/src/components/EventContainer"
import {
  StyledLogo,
  StyledLogoLink,
  StyledSidebarOpenContainer,
} from "@streamlit/app/src/components/Sidebar/styled-components"
import { AppContext } from "@streamlit/app/src/components/AppContext"

import {
  StyledAppViewBlockContainer,
  StyledAppViewBlockSpacer,
  StyledAppViewContainer,
  StyledAppViewMain,
  StyledBottomBlockContainer,
  StyledEventBlockContainer,
  StyledIFrameResizerAnchor,
  StyledInnerBottomContainer,
  StyledSidebarBlockContainer,
  StyledStickyBottomContainer,
} from "./styled-components"
import ScrollToBottomContainer from "./ScrollToBottomContainer"

export interface AppViewProps {
  elements: AppRoot

  endpoints: StreamlitEndpoints

  sendMessageToHost: (message: IGuestToHostMessage) => void

  // The unique ID for the most recent script run.
  scriptRunId: string

  scriptRunState: ScriptRunState

  widgetMgr: WidgetStateManager

  uploadClient: FileUploadClient

  // Disable the widgets when not connected to the server.
  widgetsDisabled: boolean

  componentRegistry: ComponentRegistry

  formsData: FormsData

  appLogo: Logo | null

  appPages: IAppPage[]

  navSections: string[]

  onPageChange: (pageName: string) => void

  currentPageScriptHash: string

  hideSidebarNav: boolean

  expandSidebarNav: boolean
}

/**
 * Renders a Streamlit app.
 */
function AppView(props: AppViewProps): ReactElement {
  const {
    elements,
    scriptRunId,
    scriptRunState,
    widgetMgr,
    widgetsDisabled,
    uploadClient,
    componentRegistry,
    formsData,
    appLogo,
    appPages,
    navSections,
    onPageChange,
    currentPageScriptHash,
    expandSidebarNav,
    hideSidebarNav,
    sendMessageToHost,
    endpoints,
  } = props

  React.useEffect(() => {
    const listener = (): void => {
      sendMessageToHost({
        type: "UPDATE_HASH",
        hash: window.location.hash,
      })
    }
    window.addEventListener("hashchange", listener, false)
    return () => window.removeEventListener("hashchange", listener, false)
  }, [sendMessageToHost])

  const {
    wideMode,
    initialSidebarState,
    embedded,
    showPadding,
    disableScrolling,
    showToolbar,
    showColoredLine,
    sidebarChevronDownshift,
  } = React.useContext(AppContext)

  const { addScriptFinishedHandler, removeScriptFinishedHandler } =
    React.useContext(LibContext)

  const layout = wideMode ? "wide" : "narrow"
  const hasSidebarElements = !elements.sidebar.isEmpty
  const hasEventElements = !elements.event.isEmpty
  const hasBottomElements = !elements.bottom.isEmpty

  const [showSidebarOverride, setShowSidebarOverride] = React.useState(false)

  const showSidebar =
    hasSidebarElements ||
    (!hideSidebarNav && appPages.length > 1) ||
    showSidebarOverride

  React.useEffect(() => {
    // Handle sidebar flicker/unmount with MPA & hideSidebarNav
    if (showSidebar && hideSidebarNav && !showSidebarOverride) {
      setShowSidebarOverride(true)
    }
  }, [showSidebar, hideSidebarNav, showSidebarOverride])

  const scriptFinishedHandler = React.useCallback(() => {
    // Check at end of script run if no sidebar elements
    if (!hasSidebarElements && showSidebarOverride) {
      setShowSidebarOverride(false)
    }
  }, [hasSidebarElements, showSidebarOverride])

  React.useEffect(() => {
    addScriptFinishedHandler(scriptFinishedHandler)
    return () => {
      removeScriptFinishedHandler(scriptFinishedHandler)
    }
  }, [
    scriptFinishedHandler,
    addScriptFinishedHandler,
    removeScriptFinishedHandler,
  ])

  const renderLogo = (appLogo: Logo): ReactElement => {
    const displayImage = appLogo.iconImage ? appLogo.iconImage : appLogo.image
    const source = endpoints.buildMediaURL(displayImage)

    const logo = (
      <StyledLogo
        src={source}
        size={appLogo.size}
        alt="Logo"
        className="stLogo"
        data-testid="stLogo"
      />
    )

    if (appLogo.link) {
      return (
        <StyledLogoLink
          href={appLogo.link}
          target="_blank"
          rel="noreferrer"
          data-testid="stLogoLink"
        >
          {logo}
        </StyledLogoLink>
      )
    }
    return logo
  }

  // Activate scroll to bottom whenever there are bottom elements:
  const Component = hasBottomElements
    ? ScrollToBottomContainer
    : StyledAppViewMain

  const renderBlock = (node: BlockNode): ReactElement => (
    <VerticalBlock
      node={node}
      endpoints={endpoints}
      scriptRunId={scriptRunId}
      scriptRunState={scriptRunState}
      widgetMgr={widgetMgr}
      widgetsDisabled={widgetsDisabled}
      uploadClient={uploadClient}
      componentRegistry={componentRegistry}
      formsData={formsData}
    />
  )

  // The tabindex is required to support scrolling by arrow keys.
  return (
    <StyledAppViewContainer
      className="stAppViewContainer appview-container"
      data-testid="stAppViewContainer"
      data-layout={layout}
    >
      {showSidebar && (
        <Profiler id="Sidebar">
          <ThemedSidebar
            endpoints={endpoints}
            initialSidebarState={initialSidebarState}
            appLogo={appLogo}
            appPages={appPages}
            navSections={navSections}
            hasElements={hasSidebarElements}
            onPageChange={onPageChange}
            currentPageScriptHash={currentPageScriptHash}
            hideSidebarNav={hideSidebarNav}
            expandSidebarNav={expandSidebarNav}
          >
            <StyledSidebarBlockContainer>
              {renderBlock(elements.sidebar)}
            </StyledSidebarBlockContainer>
          </ThemedSidebar>
        </Profiler>
      )}
      {!showSidebar && appLogo && (
        <StyledSidebarOpenContainer
          chevronDownshift={sidebarChevronDownshift}
          data-testid="stSidebarCollapsedControl"
        >
          {renderLogo(appLogo)}
        </StyledSidebarOpenContainer>
      )}
      <Component
        tabIndex={0}
        isEmbedded={embedded}
        disableScrolling={disableScrolling}
        className="stMain"
        data-testid="stMain"
      >
        <Profiler id="Main">
          <StyledAppViewBlockContainer
            className="stMainBlockContainer block-container"
            data-testid="stMainBlockContainer"
            isWideMode={wideMode}
            showPadding={showPadding}
            addPaddingForHeader={showToolbar || showColoredLine}
            hasBottom={hasBottomElements}
            isEmbedded={embedded}
            hasSidebar={showSidebar}
          >
            {renderBlock(elements.main)}
          </StyledAppViewBlockContainer>
        </Profiler>
        {/* Anchor indicates to the iframe resizer that this is the lowest
        possible point to determine height. But we don't add an anchor if there is
        a bottom container in the app, since those two aspects don't work
        well together. */}
        {!hasBottomElements && (
          <StyledIFrameResizerAnchor
            data-testid="stAppIframeResizerAnchor"
            data-iframe-height
          />
        )}
        {hasBottomElements && (
          <Profiler id="Bottom">
            {/* We add spacing here to make sure that the sticky bottom is
           always pinned the bottom. Using sticky layout here instead of
           absolut / fixed is a trick to automatically account for the bottom
           height in the scroll area. Thereby, the bottom container will never
           cover something if you scroll to the end.*/}
            <StyledAppViewBlockSpacer />
            <StyledStickyBottomContainer
              className="stBottom"
              data-testid="stBottom"
            >
              <StyledInnerBottomContainer>
                <StyledBottomBlockContainer
                  data-testid="stBottomBlockContainer"
                  isWideMode={wideMode}
                  showPadding={showPadding}
                >
                  {renderBlock(elements.bottom)}
                </StyledBottomBlockContainer>
              </StyledInnerBottomContainer>
            </StyledStickyBottomContainer>
          </Profiler>
        )}
      </Component>
      {hasEventElements && (
        <Profiler id="Event">
          <EventContainer scriptRunId={elements.event.scriptRunId}>
            <StyledEventBlockContainer
              className="stEvent"
              data-testid="stEvent"
            >
              {renderBlock(elements.event)}
            </StyledEventBlockContainer>
          </EventContainer>
        </Profiler>
      )}
    </StyledAppViewContainer>
  )
}

export default AppView
