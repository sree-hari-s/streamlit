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

import base64
import json
import threading
from unittest.mock import MagicMock, patch

from parameterized import parameterized

import streamlit as st
from streamlit.errors import StreamlitAPIException, StreamlitAuthError
from streamlit.runtime.forward_msg_queue import ForwardMsgQueue
from streamlit.runtime.fragment import MemoryFragmentStorage
from streamlit.runtime.pages_manager import PagesManager
from streamlit.runtime.scriptrunner import (
    ScriptRunContext,
    add_script_run_ctx,
    get_script_run_ctx,
)
from streamlit.runtime.state import SafeSessionState, SessionState
from tests.delta_generator_test_case import DeltaGeneratorTestCase

SECRETS_MOCK = {
    "redirect_uri": "http://localhost:8501/oauth2callback",
    "cookie_secret": "test_cookie_secret",
    "google": {
        "client_id": "CLIENT_ID",
        "client_secret": "CLIENT_SECRET",
        "server_metadata_url": "https://accounts.google.com/.well-known/openid-configuration",
    },
    "microsoft": {
        "client_id": "CLIENT_ID",
        "client_secret": "CLIENT_SECRET",
        "server_metadata_url": "https://login.microsoftonline.com/common/v2.0/.well-known/openid-configuration",
    },
    "auth0": {
        "client_id": "CLIENT_ID",
        "client_secret": "CLIENT_SECRET",
        "server_metadata_url": "https://YOUR_DOMAIN/.well-known/openid-configuration",
    },
}


class UserInfoProxyTest(DeltaGeneratorTestCase):
    """Test UserInfoProxy."""

    def test_user_email_attr(self):
        """Test that `st.user.email` returns user info from ScriptRunContext"""
        self.assertEqual(st.experimental_user.email, "test@example.com")

    def test_user_email_key(self):
        self.assertEqual(st.experimental_user["email"], "test@example.com")

    def test_user_non_existing_attr(self):
        """Test that an error is raised when called non existed attr."""
        with self.assertRaises(AttributeError):
            st.write(st.experimental_user.attribute)

    def test_user_non_existing_key(self):
        """Test that an error is raised when called non existed key."""
        with self.assertRaises(KeyError):
            st.write(st.experimental_user["key"])

    def test_user_cannot_be_modified_existing_key(self):
        """
        Test that an error is raised when try to assign new value to existing key.
        """
        with self.assertRaises(StreamlitAPIException) as e:
            st.experimental_user["email"] = "NEW_VALUE"

        self.assertEqual(str(e.exception), "st.experimental_user cannot be modified")

    def test_user_cannot_be_modified_new_key(self):
        """
        Test that an error is raised when try to assign new value to new key.
        """
        with self.assertRaises(StreamlitAPIException) as e:
            st.experimental_user["foo"] = "bar"

        self.assertEqual(str(e.exception), "st.experimental_user cannot be modified")

    def test_user_cannot_be_modified_existing_attr(self):
        """
        Test that an error is raised when try to assign new value to existing attr.
        """
        with self.assertRaises(StreamlitAPIException) as e:
            st.experimental_user.email = "bar"

        self.assertEqual(str(e.exception), "st.experimental_user cannot be modified")

    def test_user_cannot_be_modified_new_attr(self):
        """
        Test that an error is raised when try to assign new value to new attr.
        """
        with self.assertRaises(StreamlitAPIException) as e:
            st.experimental_user.foo = "bar"

        self.assertEqual(str(e.exception), "st.experimental_user cannot be modified")

    def test_user_len(self):
        self.assertEqual(len(st.experimental_user), 1)

    def test_st_user_reads_from_context_(self):
        """Test that st.user reads information from current ScriptRunContext
        And after ScriptRunContext changed, it returns new email
        """
        orig_report_ctx = get_script_run_ctx()

        forward_msg_queue = ForwardMsgQueue()

        try:
            add_script_run_ctx(
                threading.current_thread(),
                ScriptRunContext(
                    session_id="test session id",
                    _enqueue=forward_msg_queue.enqueue,
                    query_string="",
                    session_state=SafeSessionState(SessionState(), lambda: None),
                    uploaded_file_mgr=None,
                    main_script_path="",
                    user_info={"email": "something@else.com"},
                    fragment_storage=MemoryFragmentStorage(),
                    pages_manager=PagesManager(""),
                ),
            )

            self.assertEqual(st.experimental_user.email, "something@else.com")
        except Exception as e:
            raise e
        finally:
            add_script_run_ctx(threading.current_thread(), orig_report_ctx)


@patch(
    "streamlit.auth_util.secrets_singleton",
    MagicMock(
        load_if_toml_exists=MagicMock(return_value=True),
        get=MagicMock(return_value=SECRETS_MOCK),
    ),
)
class UserInfoAuthTest(DeltaGeneratorTestCase):
    """Test UserInfoProxy Auth functionality."""

    @parameterized.expand(["google", "microsoft", "auth0"])
    def test_user_login(self, provider):
        """Test that st.login sends correct proto message."""
        st.login(provider)

        c = self.get_message_from_queue().auth_redirect

        assert c.url.startswith("/auth/login?provider=")

        jwt_token = c.url.split("=")[1]
        raw_payload = jwt_token.split(".")[1]
        parsed_payload = json.loads(base64.b64decode(raw_payload + "==="))

        assert parsed_payload["provider"] == provider

    def test_user_login_with_invalid_provider(self):
        """Test that st.login raise exception for invalid provider."""
        with self.assertRaises(StreamlitAuthError) as ex:
            st.login("invalid_provider")

        assert (
            "Authentication credentials in `.streamlit/secrets.toml` are missing for the "
            'authentication provider "invalid_provider". Please check your configuration.'
        ) == str(ex.exception)

    def test_user_login_redirect_uri_missing(self):
        """Tests that an error is raised if the redirect uri is missing"""
        with patch(
            "streamlit.auth_util.secrets_singleton",
            MagicMock(
                load_if_toml_exists=MagicMock(return_value=True),
                get=MagicMock(return_value={"google": {}}),
            ),
        ):
            with self.assertRaises(StreamlitAuthError) as ex:
                st.login("google")

            assert """Authentication credentials in `.streamlit/secrets.toml` are missing the
            "redirect_uri" key. Please check your configuration.""" == str(ex.exception)

    def test_user_login_cookie_secret_missing(self):
        """Tests that an error is raised if the cookie secret is missing in secrets.toml"""
        with patch(
            "streamlit.auth_util.secrets_singleton",
            MagicMock(
                load_if_toml_exists=MagicMock(return_value=True),
                get=MagicMock(
                    return_value={
                        "redirect_uri": "http://localhost:8501/oauth2callback",
                        "google": {},
                    }
                ),
            ),
        ):
            with self.assertRaises(StreamlitAuthError) as ex:
                st.login("google")

            assert """Authentication credentials in `.streamlit/secrets.toml` are missing the
            "cookie_secret" key. Please check your configuration.""" == str(
                ex.exception
            )

    def test_user_login_required_fields_missing(self):
        """Tests that an error is raised if the required fields are missing"""
        with patch(
            "streamlit.auth_util.secrets_singleton",
            MagicMock(
                load_if_toml_exists=MagicMock(return_value=True),
                get=MagicMock(
                    return_value={
                        "redirect_uri": "http://localhost:8501/oauth2callback",
                        "cookie_secret": "test_cookie_secret",
                        "google": {},
                    }
                ),
            ),
        ):
            with self.assertRaises(StreamlitAuthError) as ex:
                st.login("google")

            assert (
                "Authentication credentials in `.streamlit/secrets.toml` for the "
                'authentication provider "google" are missing the following keys: '
                "['client_id', 'client_secret', 'server_metadata_url']. Please check your "
                "configuration."
            ) == str(ex.exception)

    def test_user_logout(self):
        """Test that st.logout sends correct proto message."""
        st.logout()

        c = self.get_message_from_queue().auth_redirect

        assert c.url.startswith("/auth/logout")
