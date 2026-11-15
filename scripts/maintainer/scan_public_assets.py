"""Inspect public files, nested archives, USD layers and image metadata for leaks.

USD data identities (jenga:* keys and jenga01 asset names) are intentionally
allowed. No entire file, member, prim or JSON subtree is exempted.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import ipaddress
import json
import lzma
import re
import subprocess
import tarfile
import tempfile
import zipfile
from pathlib import Path

# Spell enforcement tokens as patterns so this source can scan itself.
PATTERNS = {
    "absolute-path": re.compile(
        r"(?<![\w/])/(?:mnt|na[s]|home|root|Users|data|tmp)(?:/[^\s\"'<>]*)?(?=$|[\s\"'<>])"
        r"|\b[A-Za-z]:[\\/][^\s\"'<>]+"
    ),
    "internal-host": re.compile(r"aliy[u]n|\bjenga[-_]sim(?!_v1\b)|\bNA[S]\b", re.I),
    "internal-tool": re.compile(
        r"replay_runner_isaac[_]sim|jenga[-]sim[-]bench|tools/v[3]\b|redo[_]claude", re.I
    ),
    "email": re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.I),
}
PERSONAL_TOKEN_SHA256 = frozenset(
    {
        "cd9413036f28d6be63723b79d0dd1933ba443fb0661a323efc452be543ad70f0",
        "df891f7ebc357a9d8c9e81e3bca5a80ef973ca5ee1d897dfba1b5dbd2068c6e9",
        "662ec0bb5f98f162414ec1972b613445b61099e34ecf1fdacb427a7fa6d44176",
    }
)
# Exact public contacts published in SECURITY.md and CODE_OF_CONDUCT.md.
ALLOWED_EMAILS = {
    "fixture@example.test",
    "maintainers@tower-benchmark.invalid",
    "marcomak@tower-benchmark.com",
    "challenge@tower-benchmark.com",
}
ALLOWED_IDENTITIES = ["USD jenga:* attribute names", "jenga01 and other numbered asset names"]
IPV6 = re.compile(r"(?<![\w:])(?:[0-9a-f]{0,4}:){2,}[0-9a-f]{0,4}(?:%[\w.-]+)?(?![\w:])", re.I)


def pointer(value):
    return str(value).replace("~", "~0").replace("/", "~1")


def matches(value, *, name="", context=None):
    # Share the public-tree host/IP/version/lineage policy, including its narrow
    # documentation network and package version exceptions.
    from tower_sim.bench.public_surface import violations

    result = [("public-tree", token) for token in violations(context or value, name=name)]
    if ":" in value:
        for candidate in IPV6.finditer(value):
            if (
                re.fullmatch(r"::[0-9]+", candidate.group())
                and re.search(r"[\w\]\)]\[$", value[: candidate.start()])
                and value[candidate.end() :].startswith("]")
            ):
                continue  # Python array stride notation, including in documentation.
            try:
                address = ipaddress.IPv6Address(candidate.group())
            except ipaddress.AddressValueError:
                continue
            if (
                not address.is_loopback
                and not address.is_unspecified
                and address not in ipaddress.ip_network("2001:db8::/32")
            ):
                result.append(("ip-address", candidate.group()))
    for kind, pattern in PATTERNS.items():
        if kind == "email" and "@" not in value:
            continue
        for match in pattern.finditer(value):
            if kind == "email" and match.group() in ALLOWED_EMAILS:
                continue
            if (
                kind == "email"
                and value[match.end() :].startswith(":")
                and match.group().startswith("git@")
            ):
                continue  # Git SCP-style URL, not a person's email address.
            result.append((kind, match.group()))
    for word in re.finditer(r"\b\w+\b", value):
        token = word.group()
        if hashlib.sha256(token.lower().encode("utf-8")).hexdigest() in PERSONAL_TOKEN_SHA256:
            result.append(("personal-username", token))
    return sorted(set(result))


def image_metadata_bytes(payload):
    """Retain every ancillary byte; compressed pixels are not text metadata."""
    if payload.startswith(b"\x89PNG\r\n\x1a\n"):
        offset, parts = 8, []
        while offset + 12 <= len(payload):
            size = int.from_bytes(payload[offset : offset + 4], "big")
            kind = payload[offset + 4 : offset + 8]
            end = offset + 12 + size
            if end > len(payload):
                raise ValueError("Truncated PNG chunk")
            if kind != b"IDAT":
                parts.append(payload[offset + 8 : offset + 8 + size])
            offset = end
            if kind == b"IEND":
                parts.append(payload[offset:])
                return b"\n".join(parts)
        raise ValueError("Missing PNG end marker")
    if payload.startswith(b"\xff\xd8"):
        offset, parts = 2, []
        while offset < len(payload):
            if payload[offset] != 255:
                raise ValueError("Invalid JPEG segment")
            while payload[offset] == 255:
                offset += 1
            marker = payload[offset]
            offset += 1
            if marker == 0xD9:
                parts.append(payload[offset:])
                return b"\n".join(parts)
            size = int.from_bytes(payload[offset : offset + 2], "big")
            if size < 2 or offset + size > len(payload):
                raise ValueError("Truncated JPEG segment")
            parts.append(payload[offset + 2 : offset + size])
            offset += size
            if marker == 0xDA:
                while True:
                    offset = payload.find(b"\xff", offset)
                    if offset < 0 or offset + 1 >= len(payload):
                        raise ValueError("Missing JPEG end marker")
                    following = payload[offset + 1]
                    if following == 0 or 0xD0 <= following <= 0xD7:
                        offset += 2
                    else:
                        break
        raise ValueError("Missing JPEG end marker")
    raise ValueError("Unknown image format")


class Scanner:
    def __init__(self):
        self.hits = []
        self.errors = []
        self.inspected = []
        self._seen = set()
        self.numeric_candidates = []

    def value(self, value, file, member, location):
        if isinstance(value, bytes):
            self.value(value.decode("utf-8", "replace"), file, member, location)
        elif hasattr(value, "tolist"):
            self.value(value.tolist(), file, member, location)
        elif isinstance(value, dict):
            for key, item in value.items():
                self.value(str(key), file, member, location + "/" + pointer(key) + "#key")
                self.value(item, file, member, location + "/" + pointer(key))
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                self.value(item, file, member, location + "/" + str(index))
        elif isinstance(value, str):
            # Embedded JSON (notably the calibrated station configuration) gets
            # real pointers instead of an opaque, multi-kilobyte attribute hit.
            if value.lstrip().startswith(("{", "[")):
                try:
                    nested = json.loads(value)
                except ValueError:
                    pass
                else:
                    self.value(nested, file, member, location + "#json")
                    return
            context = json.dumps({location.rsplit("/", 1)[-1]: value})
            for kind, token in matches(
                value,
                name=member or file,
                context=context if re.fullmatch(r"(?:\d{1,3}\.){3}\d{1,3}", value) else None,
            ):
                key = (file, member, location, kind, token, value)
                if key not in self._seen:
                    self._seen.add(key)
                    self.hits.append(
                        dict(
                            file=file,
                            member=member,
                            location=location,
                            kind=kind,
                            match=token,
                            value=value,
                        )
                    )

    def usd(self, payload, suffix, file, member):
        from pxr import Sdf

        with tempfile.TemporaryDirectory(prefix="tower-scan-") as temporary:
            path = Path(temporary) / ("layer" + suffix)
            path.write_bytes(payload)
            layer = Sdf.Layer.FindOrOpen(str(path))
            if layer is None:
                raise ValueError("USD layer could not be opened")
            ranges, buffers = [], set()
            paths = []
            layer.Traverse(Sdf.Path.absoluteRootPath, paths.append)
            for spec_path in paths:
                spec = (
                    layer.pseudoRoot
                    if spec_path == Sdf.Path.absoluteRootPath
                    else layer.GetObjectAtPath(spec_path)
                )
                if spec is None:
                    if not spec_path.IsTargetPath():
                        raise ValueError(f"Uninspectable USD spec: {spec_path}")
                    self.value(str(spec_path), file, member, "USD:target")
                    continue
                for field in spec.ListInfoKeys():
                    value = spec.GetInfo(field)
                    if not isinstance(value, (dict, list, tuple, str)):
                        try:
                            view = memoryview(value)
                        except TypeError:
                            value = str(value)
                        else:
                            raw = view.tobytes()
                            if raw and raw not in buffers:
                                buffers.add(raw)
                                offset = payload.find(raw)
                                if offset >= 0:
                                    ranges.append(
                                        (offset, offset + len(raw), f"{spec_path}:{field}")
                                    )
                            continue  # Numeric buffers contain no string metadata.
                    self.value(value, file, member, f"{spec_path}:{field}")
            # Export every layer, even when fields above yielded no findings.
            # This catches authored constructs not exposed as ordinary specs.
            for number, line in enumerate(layer.ExportToString().splitlines(), 1):
                if not any(mark in line for mark in ('"', "'", "@", "#")):
                    line = line.partition("=")[0]  # Numeric RHS has no string metadata.
                self.value(line, file, member, f"SdfExport:{number}")
            return ranges

    def payload(self, payload, name, file, member="", depth=0):
        if depth > 12:
            raise ValueError("Archive nesting exceeds the inspection limit")
        self.inspected.append(dict(file=file, member=member, size=len(payload)))
        self.value(name, file, member, "filename")
        suffix = Path(name).suffix.lower()
        for magic, detected in (
            (b"PXR-USDC", ".usdc"),
            (b"#usda", ".usda"),
            (b"\x89PNG\r\n\x1a\n", ".png"),
            (b"\xff\xd8", ".jpg"),
            (b"\x93NUMPY", ".npy"),
            (b"\x89HDF\r\n\x1a\n", ".hdf5"),
            (b"\xfd7zXZ\x00", ".xz"),
            (b"\x1f\x8b", ".gz"),
        ):
            if payload.startswith(magic):
                suffix = detected
                break
        if payload[257:262] == b"ustar":
            suffix = ".tar"
        numeric_ranges = []

        def child(child_name):
            return member + "!" + child_name if member else child_name

        stream = io.BytesIO(payload)
        if zipfile.is_zipfile(stream):
            with zipfile.ZipFile(stream) as archive:
                self.value(archive.comment.decode("utf-8", "replace"), file, member, "zip:comment")
                for item in archive.infolist():
                    self.value(item.filename, file, child(item.filename), "zip:name")
                    self.value(
                        item.extra.decode("utf-8", "replace"),
                        file,
                        child(item.filename),
                        "zip:extra",
                    )
                    self.value(
                        item.comment.decode("utf-8", "replace"),
                        file,
                        child(item.filename),
                        "zip:comment",
                    )
                    if not item.is_dir():
                        self.payload(
                            archive.read(item), item.filename, file, child(item.filename), depth + 1
                        )
            return
        stream.seek(0)
        if suffix in {".xz", ".gz", ".tgz", ".tar", ".bz2"}:
            try:
                archive = tarfile.open(fileobj=io.BytesIO(payload), mode="r:*")  # noqa: SIM115 -- entered below after format detection
            except tarfile.ReadError:
                if suffix == ".xz":
                    contents = lzma.decompress(payload)
                elif suffix == ".gz":
                    contents = gzip.decompress(payload)
                else:
                    raise
                self.payload(
                    contents, name.rsplit(".", 1)[0], file, child(name.rsplit(".", 1)[0]), depth + 1
                )
            else:
                with archive:
                    for item in archive.getmembers():
                        self.value(item.pax_headers, file, child(item.name), "tar:pax")
                        self.value(item.name, file, child(item.name), "tar:name")
                        self.value(item.linkname, file, child(item.name), "tar:link")
                        self.value(item.uname, file, child(item.name), "tar:uname")
                        self.value(item.gname, file, child(item.name), "tar:gname")
                        if item.isfile():
                            source = archive.extractfile(item)
                            if source is None:
                                raise ValueError("Missing regular tar member payload")
                            self.payload(
                                source.read(),
                                item.name,
                                file,
                                child(item.name),
                                depth + 1,
                            )
            return
        if suffix in {".usd", ".usdc", ".usda"}:
            numeric_ranges = self.usd(payload, suffix, file, member)
        elif suffix == ".npy":
            import numpy as np

            array = np.load(stream, allow_pickle=False)
            version = tuple(payload[6:8])
            length_size = 2 if version == (1, 0) else 4
            start = 8 + length_size + int.from_bytes(payload[8 : 8 + length_size], "little")
            if array.dtype.kind not in "SUV":
                end = start + array.nbytes
                if end > len(payload):
                    raise ValueError("Truncated numeric array")
                numeric_ranges.append((start, end, "numpy:numeric-array"))
        elif suffix in {".h5", ".hdf5"}:
            import h5py

            with h5py.File(stream, "r") as handle:

                def inspect(name, obj):
                    self.value(name, file, member, "hdf5:name")
                    for key, value in obj.attrs.items():
                        self.value(str(key), file, member, f"hdf5:{name}/@{key}#key")
                        self.value(value, file, member, f"hdf5:{name}/@{key}")
                    if isinstance(obj, h5py.Dataset):
                        if obj.dtype.kind in "SUV" or obj.dtype.hasobject:
                            self.value(obj[()], file, member, f"hdf5:{name}")
                        elif obj.chunks:
                            for index in range(obj.id.get_num_chunks()):
                                chunk = obj.id.get_chunk_info(index)
                                numeric_ranges.append(
                                    (
                                        chunk.byte_offset,
                                        chunk.byte_offset + chunk.size,
                                        f"hdf5:{name}",
                                    )
                                )
                        elif obj.id.get_offset() is not None:
                            start = obj.id.get_offset()
                            numeric_ranges.append(
                                (start, start + obj.id.get_storage_size(), f"hdf5:{name}")
                            )

                inspect("/", handle)
                handle.visititems(inspect)
        elif suffix in {".png", ".jpg", ".jpeg"}:
            from PIL import Image

            with Image.open(stream) as picture:
                self.value({k: str(v) for k, v in picture.info.items()}, file, member, "image:info")
                exif = picture.getexif()
                self.value({str(k): str(v) for k, v in exif.items()}, file, member, "image:exif")
                for tag in (34665, 34853, 40965):
                    if tag in exif:
                        self.value(
                            {str(k): str(v) for k, v in exif.get_ifd(tag).items()},
                            file,
                            member,
                            f"image:exif:{tag}",
                        )
        else:
            try:
                text = payload.decode("utf-8-sig")
            except UnicodeError:
                text = None
            if text is not None:
                if suffix == ".json":
                    self.value(json.loads(text), file, member, "")
                elif suffix in {".yaml", ".yml"}:
                    import yaml

                    for index, document in enumerate(yaml.safe_load_all(text)):
                        self.value(document, file, member, f"yaml:{index}")
                # Scan comments, scalar text, URDF/xacro, Markdown and code too.
                for number, line in enumerate(text.splitlines(), 1):
                    self.value(line, file, member, f"line:{number}")
                return
        if suffix in {".png", ".jpg", ".jpeg"}:
            payload = image_metadata_bytes(payload)
        # Equivalent to the printable runs from strings(1), without executing an
        # external command on untrusted filenames. Includes compressed image and
        # binary metadata that format-specific readers may not expose.
        for match in re.finditer(rb"[\x20-\x7e]{4,}", payload):
            value = match.group().decode("ascii")
            containing = next(
                (
                    row
                    for row in numeric_ranges
                    if row[0] <= match.start() and match.end() <= row[1]
                ),
                None,
            )
            if containing is not None:
                if matches(value, name=member or file):
                    self.numeric_candidates.append(
                        dict(
                            file=file,
                            member=member,
                            offset=match.start(),
                            value=value,
                            numeric_region=containing,
                        )
                    )
            else:
                self.value(value, file, member, f"strings:{match.start()}")

    def scan(self, path, name=None):
        path = Path(path)
        name = name or path.name
        try:
            self.payload(path.read_bytes(), name, name)
        except Exception as error:  # noqa: BLE001 -- unreadable input makes the gate fail
            self.errors.append(dict(file=name, error=f"{type(error).__name__}: {error}"))

    def report(self):
        return dict(
            schema="tower-public-asset-scan-v1",
            hits=self.hits,
            errors=self.errors,
            inspected=self.inspected,
            numeric_byte_candidates=self.numeric_candidates,
            allowed_identities=ALLOWED_IDENTITIES,
            passed=not self.hits and not self.errors,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", type=Path, nargs="+")
    parser.add_argument("--tracked", action="store_true", help="Only tracked repository files")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    scanner = Scanner()
    for root in args.roots:
        if root.is_file():
            scanner.scan(root)
        elif args.tracked:
            names = (
                subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode().split("\0")
            )
            for name in names:
                if name and (root / name).is_file():
                    scanner.scan(root / name, name)
        elif not root.is_dir():
            scanner.errors.append(dict(file=str(root), error="Scan root does not exist"))
        else:
            for path in sorted(root.rglob("*")):
                if path.is_file():
                    scanner.scan(path, path.relative_to(root).as_posix())
    report = scanner.report()
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    else:
        print(json.dumps(report, indent=2))
    print(f"Public asset scan: {len(scanner.hits)} hits, {len(scanner.errors)} errors")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
