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

"""Tests that the distributed skill trees stay byte-identical."""

from __future__ import annotations

from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).parent.parent
_PLUGIN_SKILLS = _REPO_ROOT / "plugins" / "mcp-gemini-search" / "skills"
_MIRROR_SKILLS = {
    ".agents/skills": _REPO_ROOT / ".agents" / "skills",
    ".claude/skills": _REPO_ROOT / ".claude" / "skills",
}


def _skill_files(root: Path) -> dict[str, str]:
    return {str(path.relative_to(root)): path.read_text(encoding="utf-8") for path in sorted(root.rglob("SKILL.md"))}


@pytest.mark.parametrize("mirror", sorted(_MIRROR_SKILLS))
def test_skill_trees_stay_identical(mirror: str) -> None:
    """The same skills ship in the plugin, ``.claude``, and ``.agents`` trees.

    Nothing synchronizes the copies mechanically, so this guard turns silent
    drift — one host shipping different guidance than another — into a test
    failure naming the drifted file.
    """
    canonical = _skill_files(_PLUGIN_SKILLS)
    mirrored = _skill_files(_MIRROR_SKILLS[mirror])

    assert canonical, "no SKILL.md files found under plugins/mcp-gemini-search/skills"
    assert mirrored.keys() == canonical.keys()
    for rel, content in canonical.items():
        assert mirrored[rel] == content, f"{mirror}/{rel} drifted from the plugin copy"
