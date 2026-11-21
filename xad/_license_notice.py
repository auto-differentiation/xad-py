##############################################################################
#
#  XAD Python bindings
#
#  This file is part of XAD's Python bindings, a comprehensive library for
#  automatic differentiation.
#
#  Copyright (C) 2010-2026 Xcelerit Computing Ltd.
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero General Public License as published
#  by the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Affero General Public License for more details.
#
#  You should have received a copy of the GNU Affero General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
##############################################################################

"""The license notice, shown once per Python environment"""

import hashlib
import os
import sys

_NOTICE = (
    "XAD is licensed under the AGPL. "
    "For production use, XAD is available under an annual subscription:\n"
    "https://www.xcelerit.com/xad/licensing/"
)

# one notice per process
_shown = False


def _cache_dir() -> str:
    if sys.platform == "win32":
        return os.environ.get("LOCALAPPDATA") or os.path.expanduser("~/AppData/Local")
    if sys.platform == "darwin":
        return os.path.expanduser("~/Library/Caches")
    return os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")


def _marker() -> str | None:
    cache_dir = _cache_dir()
    if not os.path.isabs(cache_dir):
        return None  # no home directory: keep no state rather than write relative to the cwd
    env_key = hashlib.sha256(os.fsencode(sys.prefix)).hexdigest()[:16]
    return os.path.join(cache_dir, "xad", f"license-notice-{env_key}")


def _show() -> None:
    marker = _marker()
    if marker and os.path.exists(marker):
        return
    sys.stderr.write(_NOTICE + "\n")
    if marker:
        os.makedirs(os.path.dirname(marker), exist_ok=True)
        open(marker, "w").close()


def show_once() -> None:
    """Print the license notice to stderr, once per Python environment"""
    global _shown
    if _shown:
        return
    _shown = True
    try:
        _show()
    except Exception:
        pass  # the notice never fails an import; without a marker it shows again next time
