# Copyright 2026 The mcp-gemini-search Authors.
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

"""Package version derived from installed distribution metadata."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("mcp-gemini-search")
except PackageNotFoundError:
    # Running from a source tree that is not installed; mirror the
    # uv-dynamic-versioning fallback-version.
    __version__ = "0.0.0"
