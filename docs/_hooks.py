# SPDX-License-Identifier: Apache-2.0
"""Keep source-tree links usable in the hosted documentation."""

import re
from pathlib import Path
from urllib.parse import urlsplit


def on_page_markdown(markdown, *, page, config, files):
    root = Path(config["docs_dir"]).parent
    source = Path(page.file.abs_src_path)

    def replace(match):
        link = match.group(2)
        url = urlsplit(link)
        if url.scheme or url.netloc or not url.path or url.path.startswith("/"):
            return match.group(0)
        target = (source.parent / url.path).resolve()
        if target.is_relative_to(Path(config["docs_dir"]).resolve()):
            return match.group(0)
        if not target.is_relative_to(root.resolve()) or not target.exists():
            return match.group(0)
        relative = target.relative_to(root.resolve()).as_posix()
        suffix = "#" + url.fragment if url.fragment else ""
        return f"{match.group(1)}https://github.com/tower-benchmark/TOWER/blob/main/{relative}{suffix})"

    return re.sub(r"(!?\[[^]]*\]\()([^\s)]+)\)", replace, markdown)
