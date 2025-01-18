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

import mimetypes
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import tornado.httpserver
import tornado.testing
import tornado.web
import tornado.websocket

from streamlit.web.server import Server
from streamlit.web.server.app_static_file_handler import (
    MAX_APP_STATIC_FILE_SIZE,
    AppStaticFileHandler,
)


@unittest.skipIf(
    "win32" in sys.platform,
    "Most windows installs do not support symlinks except as admin",
)
class AppStaticFileHandlerTest(tornado.testing.AsyncHTTPTestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory(dir=os.getcwd())
        self._tmpfile = tempfile.NamedTemporaryFile(dir=self._tmpdir.name, delete=False)
        self._tmp_js_file = tempfile.NamedTemporaryFile(
            dir=self._tmpdir.name, suffix="script.js", delete=False
        )
        self._tmp_webp_file = tempfile.NamedTemporaryFile(
            dir=self._tmpdir.name, suffix="file.webp", delete=False
        )
        self._tmp_png_image_file = tempfile.NamedTemporaryFile(
            dir=self._tmpdir.name, suffix="image.png", delete=False
        )
        self._tmp_pdf_document_file = tempfile.NamedTemporaryFile(
            dir=self._tmpdir.name, suffix="document.pdf", delete=False
        )
        self._tmp_webp_image_file = tempfile.NamedTemporaryFile(
            dir=self._tmpdir.name, suffix="image.webp", delete=False
        )
        self._tmp_dir_inside_static_folder = tempfile.TemporaryDirectory(
            dir=self._tmpdir.name
        )

        self._symlink_outside_directory = "symlink_outside"
        self._symlink_inside_directory = "symlink_inside"

        os.symlink(
            "/", os.path.join(self._tmpdir.name, self._symlink_outside_directory)
        )
        os.symlink(
            self._tmpfile.name,
            os.path.join(self._tmpdir.name, self._symlink_inside_directory),
        )

        self._filename = os.path.basename(self._tmpfile.name)
        self._js_filename = os.path.basename(self._tmp_js_file.name)
        self._webp_filename = os.path.basename(self._tmp_webp_file.name)
        self._png_image_filename = os.path.basename(self._tmp_png_image_file.name)
        self._pdf_document_filename = os.path.basename(self._tmp_pdf_document_file.name)
        self._webp_image_filename = os.path.basename(self._tmp_webp_image_file.name)

        super().setUp()

    def tearDown(self) -> None:
        super().tearDown()
        self._tmpdir.cleanup()

    def get_app(self):
        return tornado.web.Application(
            [
                (
                    r"/app/static/(.*)",
                    AppStaticFileHandler,
                    {"path": "%s" % self._tmpdir.name},
                )
            ]
        )

    def test_static_files_200(self):
        """Files with extensions NOT listed in app_static_file_handler.py
        `SAFE_APP_STATIC_FILE_EXTENSIONS` should have the `Content-Type` header value
        equals to `text-plain`.
        """
        responses = [
            # self._filename is file without extension
            self.fetch(f"/app/static/{self._filename}"),
            # self._js_filename is file with '.js' extension
            self.fetch(f"/app/static/{self._js_filename}"),
            # self._symlink_inside_directory is symlink to
            # self._tmpfile (inside static directory)
            self.fetch(f"/app/static/{self._symlink_inside_directory}"),
        ]
        for r in responses:
            assert r.headers["Content-Type"] == "text/plain"
            assert r.headers["X-Content-Type-Options"] == "nosniff"
            assert r.code == 200

    def test_static_png_image_200(self):
        """Files with extensions listed in app_static_file_handler.py
        `SAFE_APP_STATIC_FILE_EXTENSIONS` (e.g. png) should have the
        `Content-Type` header based on their extension.
        """
        response = self.fetch(f"/app/static/{self._png_image_filename}")

        assert response.code == 200
        assert response.headers["Content-Type"] == "image/png"
        assert response.headers["X-Content-Type-Options"] == "nosniff"

    def test_static_webp_image_200(self):
        """Files with extensions listed in app_static_file_handler.py
        `SAFE_APP_STATIC_FILE_EXTENSIONS` (e.g. webp) should have the
        `Content-Type` header based on their extension.
        """
        response = self.fetch(f"/app/static/{self._webp_image_filename}")

        assert response.code == 200
        assert response.headers["Content-Type"] == "image/webp"
        assert response.headers["X-Content-Type-Options"] == "nosniff"

    def test_static_pdf_document_200(self):
        """Files with extensions listed in app_static_file_handler.py
        `SAFE_APP_STATIC_FILE_EXTENSIONS` (e.g. pdf) should have the
        `Content-Type` header based on their extension.
        """
        response = self.fetch(f"/app/static/{self._pdf_document_filename}")

        assert response.code == 200
        assert response.headers["Content-Type"] == "application/pdf"
        assert response.headers["X-Content-Type-Options"] == "nosniff"

    @patch("os.path.getsize", MagicMock(return_value=MAX_APP_STATIC_FILE_SIZE + 1))
    def test_big_file_404(self):
        """Files with size greater than MAX_APP_STATIC_FILE_SIZE should return 404."""
        response = self.fetch(f"/app/static/{self._png_image_filename}")
        assert response.code == 404
        self.assertEqual(
            b"<html><title>404: File is too large</title>"
            b"<body>404: File is too large</body></html>",
            response.body,
        )

    def test_staticfiles_404(self):
        """Non-existent files, files outside static directory and symlinks pointing to
        files outside static directory and directories should return 404.
        """
        responses = [
            # Access to directory without trailing slash
            self.fetch("/app/static"),
            # Access to non-existent file
            self.fetch("/app/static/nonexistent.jpg"),
        ]
        for r in responses:
            assert r.code == 404
            assert (
                r.body == b"<html><title>404: Not Found</title>"
                b"<body>404: Not Found</body></html>"
            )

    def test_staticfiles_403(self):
        """files outside static directory and symlinks pointing to
        files outside static directory and directories should return 403.
        """
        responses = [
            # Access to directory with trailing slash
            self.fetch("/app/static/"),
            # Access to directory inside static folder without trailing slash
            self.fetch(f"/app/static/{self._tmp_dir_inside_static_folder.name}"),
            # Access to directory inside static folder with trailing slash
            self.fetch(f"/app/static/{self._tmp_dir_inside_static_folder.name}/"),
            # Access to file outside static directory
            self.fetch("/app/static/../test_file_outside_directory.py"),
            # Access to file outside static directory with same prefix
            self.fetch(
                f"/app/static/{self._tmpdir.name}_foo/test_file_outside_directory.py"
            ),
            # Access to symlink outside static directory
            self.fetch(f"/app/static/{self._symlink_outside_directory}"),
        ]
        for r in responses:
            assert r.code == 403
            assert (
                r.body == b"<html><title>403: Forbidden</title>"
                b"<body>403: Forbidden</body></html>"
            )

    def test_mimetype_is_overridden_by_server(self):
        """Test content type of webps are set correctly"""
        mimetypes.add_type("custom/webp", ".webp")

        r = self.fetch(f"/app/static/{self._webp_filename}")
        assert r.headers["Content-Type"] == "custom/webp"

        Server.initialize_mimetypes()

        r = self.fetch(f"/app/static/{self._webp_filename}")
        assert r.headers["Content-Type"] == "image/webp"
