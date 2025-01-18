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

import copy
import os
import unittest
from typing import Final
from unittest.mock import MagicMock, patch

import pytest

from streamlit import config
from streamlit.elements.lib import utils
from streamlit.elements.lib.policies import (
    check_cache_replay_rules,
    check_callback_rules,
    check_fragment_path_policy,
    check_session_state_rules,
    check_widget_policies,
)
from streamlit.errors import (
    StreamlitAPIException,
    StreamlitValueAssignmentNotAllowedError,
)
from streamlit.runtime.scriptrunner_utils.script_run_context import in_cached_function

_KEY: Final = "the key"


class ElementPoliciesTest(unittest.TestCase):
    pass


class CheckCallbackRulesTest(ElementPoliciesTest):
    @patch("streamlit.elements.lib.policies.is_in_form", MagicMock(return_value=False))
    @patch("streamlit.runtime.Runtime.exists", MagicMock(return_value=True))
    def test_check_callback_rules_not_in_form(self):
        check_callback_rules(MagicMock(), lambda x: x)

    @patch("streamlit.elements.lib.policies.is_in_form", MagicMock(return_value=True))
    @patch("streamlit.runtime.Runtime.exists", MagicMock(return_value=True))
    def test_check_callback_rules_in_form(self):
        check_callback_rules(MagicMock(), None)

    @patch("streamlit.elements.lib.policies.is_in_form", MagicMock(return_value=True))
    @patch("streamlit.runtime.Runtime.exists", MagicMock(return_value=True))
    def test_check_callback_rules_error(self):
        with pytest.raises(StreamlitAPIException) as e:
            check_callback_rules(MagicMock(), lambda x: x)

        assert "is not allowed." in str(e.value)


class CheckSessionStateRules(ElementPoliciesTest):
    @patch("streamlit.warning")
    def test_check_session_state_rules_no_key(self, patched_st_warning):
        check_session_state_rules(5, key=None)

        patched_st_warning.assert_not_called()

    @patch("streamlit.runtime.Runtime.exists", MagicMock(return_value=True))
    @patch("streamlit.elements.lib.policies.get_session_state")
    @patch("streamlit.warning")
    def test_check_session_state_rules_no_val(
        self, patched_st_warning, patched_get_session_state
    ):
        mock_session_state = MagicMock()
        mock_session_state.is_new_state_value.return_value = True
        patched_get_session_state.return_value = mock_session_state

        check_session_state_rules(None, key=_KEY)

        patched_st_warning.assert_not_called()

    @patch("streamlit.runtime.Runtime.exists", MagicMock(return_value=True))
    @patch("streamlit.elements.lib.policies.get_session_state")
    @patch("streamlit.warning")
    def test_check_session_state_rules_no_state_val(
        self, patched_st_warning, patched_get_session_state
    ):
        mock_session_state = MagicMock()
        mock_session_state.is_new_state_value.return_value = False
        patched_get_session_state.return_value = mock_session_state

        check_session_state_rules(5, key=_KEY)

        patched_st_warning.assert_not_called()

    @patch("streamlit.runtime.Runtime.exists", MagicMock(return_value=True))
    @patch("streamlit.elements.lib.policies.get_session_state")
    @patch("streamlit.warning")
    def test_check_session_state_rules_hide_warning_if_state_duplication_disabled(
        self, patched_st_warning, patched_get_session_state
    ):
        config._set_option("global.disableWidgetStateDuplicationWarning", True, "test")

        mock_session_state = MagicMock()
        mock_session_state.is_new_state_value.return_value = True
        patched_get_session_state.return_value = mock_session_state

        check_session_state_rules(5, key=_KEY)

        patched_st_warning.assert_not_called()

    @patch("streamlit.runtime.Runtime.exists", MagicMock(return_value=True))
    @patch("streamlit.elements.lib.policies.get_session_state")
    def test_check_session_state_rules_writes_not_allowed(
        self, patched_get_session_state
    ):
        mock_session_state = MagicMock()
        mock_session_state.is_new_state_value.return_value = True
        patched_get_session_state.return_value = mock_session_state

        with self.assertRaises(StreamlitValueAssignmentNotAllowedError):
            check_session_state_rules(5, key=_KEY, writes_allowed=False)


class SpecialSessionStatesTest(ElementPoliciesTest):
    SECTION_DESCRIPTIONS = copy.deepcopy(config._section_descriptions)
    CONFIG_OPTIONS = copy.deepcopy(config._config_options)

    def setUp(self):
        self.patches = [
            patch.object(
                config,
                "_section_descriptions",
                new=copy.deepcopy(SpecialSessionStatesTest.SECTION_DESCRIPTIONS),
            ),
            patch.object(
                config,
                "_config_options",
                new=copy.deepcopy(SpecialSessionStatesTest.CONFIG_OPTIONS),
            ),
            patch.dict(os.environ),
        ]

        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()

        config._delete_option("_test.tomlTest")

    @patch("streamlit.runtime.Runtime.exists", MagicMock(return_value=True))
    @patch("streamlit.elements.lib.policies.get_session_state")
    @patch("streamlit.warning")
    def test_check_session_state_rules_prints_warning(
        self, patched_st_warning, patched_get_session_state
    ):
        mock_session_state = MagicMock()
        mock_session_state.is_new_state_value.return_value = True
        patched_get_session_state.return_value = mock_session_state
        # Reset globale flag:
        utils._shown_default_value_warning = False

        check_session_state_rules(5, key=_KEY)

        patched_st_warning.assert_called_once()
        args, _ = patched_st_warning.call_args
        warning_msg = args[0]
        assert 'The widget with key "the key"' in warning_msg


class CheckCacheReplayTest(ElementPoliciesTest):
    @patch("streamlit.exception")
    def test_cache_replay_rules_succeeds(self, patched_st_exception):
        check_cache_replay_rules()
        patched_st_exception.assert_not_called()

    @patch("streamlit.exception")
    def test_cache_replay_rules_fails(self, patched_st_exception):
        in_cached_function.set(True)
        check_cache_replay_rules()
        patched_st_exception.assert_called()
        # Reset the global flag to avoid affecting other tests
        in_cached_function.set(False)


class FragmentCannotWriteToOutsidePathTest(unittest.TestCase):  #
    def setUp(self):
        ctx = MagicMock()
        ctx.current_fragment_id = "my_fragment_id"
        ctx.current_fragment_delta_path = [0, 1, 2]
        self.ctx = ctx

    @patch("streamlit.elements.lib.policies.get_script_run_ctx")
    def test_when_element_delta_path_length_is_smaller_than_parent_then_raise(
        self, patched_get_script_run_ctx: MagicMock
    ):
        patched_get_script_run_ctx.return_value = self.ctx
        dg = MagicMock()
        dg._active_dg._cursor = MagicMock()
        dg._active_dg._cursor.delta_path = [0, 1]
        with self.assertRaises(StreamlitAPIException):
            check_fragment_path_policy(dg)

    @patch("streamlit.elements.lib.policies.get_script_run_ctx")
    def test_when_element_delta_path_is_not_in_parent_delta_path_then_raise(
        self, patched_get_script_run_ctx: MagicMock
    ):
        patched_get_script_run_ctx.return_value = self.ctx
        dg = MagicMock()
        dg._active_dg._cursor = MagicMock()
        dg._active_dg._cursor.delta_path = [0, 2, 0]
        with self.assertRaises(StreamlitAPIException):
            check_fragment_path_policy(dg)

    @patch("streamlit.elements.lib.policies.get_script_run_ctx")
    def test_when_element_delta_path_is_in_parent_delta_path_then_dont_raise(
        self, patched_get_script_run_ctx: MagicMock
    ):
        patched_get_script_run_ctx.return_value = self.ctx
        dg = MagicMock()
        dg._active_dg._cursor = MagicMock()
        dg._active_dg._cursor.delta_path = [0, 1, 2, 0]
        check_fragment_path_policy(dg)


@patch("streamlit.elements.lib.policies.check_session_state_rules")
@patch("streamlit.elements.lib.policies.check_callback_rules")
@patch("streamlit.elements.lib.policies.check_cache_replay_rules")
@patch("streamlit.elements.lib.policies.check_fragment_path_policy")
class CheckWidget(ElementPoliciesTest):
    def test_all_relevant_policies_are_called(
        self,
        patched_check_fragment_path_policy,
        patched_check_cache_replay_rules,
        patched_check_callback_rules,
        patched_check_session_state_rules,
    ):
        def on_change():
            """Noop"""
            pass

        dg = MagicMock()
        key = "my_key"
        default_value = 5
        check_widget_policies(dg, key, on_change, default_value=default_value)
        patched_check_fragment_path_policy.assert_called_once()
        patched_check_cache_replay_rules.assert_called_once()
        patched_check_callback_rules.assert_called_once_with(dg, on_change)
        patched_check_session_state_rules.assert_called_once_with(
            default_value=default_value, key=key, writes_allowed=True
        )

    def test_check_callback_rules_is_not_called(
        self,
        patched_check_fragment_path_policy,
        patched_check_cache_replay_rules,
        patched_check_callback_rules,
        patched_check_session_state_rules,
    ):
        check_widget_policies(
            MagicMock(), None, None, enable_check_callback_rules=False
        )
        patched_check_fragment_path_policy.assert_called_once()
        patched_check_cache_replay_rules.assert_called_once()
        patched_check_callback_rules.assert_not_called()
        patched_check_session_state_rules.assert_called_once()

    def test_writes_allowed_can_be_disabled(
        self,
        patched_check_fragment_path_policy,
        patched_check_cache_replay_rules,
        patched_check_callback_rules,
        patched_check_session_state_rules,
    ):
        dg = MagicMock()
        key = "my_key"
        check_widget_policies(dg, key, None, writes_allowed=False)
        patched_check_fragment_path_policy.assert_called_once()
        patched_check_cache_replay_rules.assert_called_once()
        patched_check_callback_rules.assert_called_once()
        patched_check_session_state_rules.assert_called_once_with(
            default_value=None, key=key, writes_allowed=False
        )
