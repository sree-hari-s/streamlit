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

import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx

ctx = get_script_run_ctx()
if ctx is None:
    import sys

    # This script is not compatible with running it in "bare" mode (e.g. `python script.py`)
    # The reason is that the mutable container is not correctly returned if
    # the runtime doesn't exist.
    print("This test script does not support bare script execution.")
    sys.exit(0)


running_status = st.status("Running status", expanded=True)
running_status.write("Doing some work...")

with st.status("Completed status", expanded=True, state="complete"):
    st.write("Hello world")

with st.status("Error status", expanded=True, state="error"):
    st.error("Oh no, something went wrong!")

with st.status("Collapsed", state="complete"):
    st.write("Hello world")

with st.status("About to change label...", state="complete") as status:
    st.write("Hello world")
    status.update(label="Changed label")

status = st.status("Without context manager", state="complete")
status.write("Hello world")
status.update(state="error", expanded=True)

with st.status("Collapse via update...", state="complete", expanded=True) as status:
    st.write("Hello world")
    status.update(label="Collapsed", expanded=False)

st.status("Empty state...", state="complete")

try:
    with st.status("Uncaught exception"):
        st.write("Hello world")
        raise Exception("Error!")
except Exception:
    pass
