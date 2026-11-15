"""Public leak gate inspects decoded content and fails closed on unreadable input."""

import io
import json
import lzma
import tarfile
import zipfile

import pytest

from scripts.maintainer.scan_public_assets import Scanner

PRIVATE = "/" + "home/operator/private.json"


def scan(tmp_path, name, payload):
    path = tmp_path / name
    path.write_bytes(payload)
    scanner = Scanner()
    scanner.scan(path)
    return scanner


@pytest.mark.parametrize("name", ["a.json", "a.yaml", "a.urdf", "a.xacro", "a.md", "a.bin"])
def test_plain_and_binary_metadata(tmp_path, name):
    result = scan(tmp_path, name, json.dumps({"source": PRIVATE}).encode())
    assert not result.errors
    assert any(PRIVATE in row["value"] for row in result.hits)


def test_nested_archive_json_pointer(tmp_path):
    tar = io.BytesIO()
    with tarfile.open(fileobj=tar, mode="w") as archive:
        payload = json.dumps({"a/b": {"source": PRIVATE}}).encode()
        item = tarfile.TarInfo("metadata.json")
        item.size = len(payload)
        archive.addfile(item, io.BytesIO(payload))
    outer = io.BytesIO()
    with zipfile.ZipFile(outer, "w") as archive:
        archive.writestr("inside.tar.xz", lzma.compress(tar.getvalue()))
    result = scan(tmp_path, "outer.zip", outer.getvalue())
    assert not result.errors
    assert any(
        row["member"] == "inside.tar.xz!metadata.json" and row["location"] == "/a~1b/source"
        for row in result.hits
    )


def test_png_text_and_jpeg_exif(tmp_path):
    from PIL import Image, PngImagePlugin

    picture = Image.new("RGB", (2, 2))
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("source", PRIVATE, zip=True)
    png = io.BytesIO()
    picture.save(png, format="PNG", pnginfo=metadata)
    assert any(
        row["location"] == "image:info/source"
        for row in scan(tmp_path, "a.png", png.getvalue()).hits
    )
    exif = picture.getexif()
    exif[270] = PRIVATE
    jpeg = io.BytesIO()
    picture.save(jpeg, format="JPEG", exif=exif)
    assert any(
        row["location"] == "image:exif/270" for row in scan(tmp_path, "a.jpg", jpeg.getvalue()).hits
    )


def test_usdz_inspects_every_layer(tmp_path):
    from pxr import Sdf

    layer = Sdf.Layer.CreateNew(str(tmp_path / "payload.usdc"))
    layer.documentation = PRIVATE
    prim = Sdf.CreatePrimInLayer(layer, "/World")
    attr = Sdf.AttributeSpec(prim, "metadata", Sdf.ValueTypeNames.String)
    attr.default = json.dumps({"nested": PRIVATE})
    layer.Save()
    container = io.BytesIO()
    with zipfile.ZipFile(container, "w") as archive:
        archive.writestr("first.usda", '#usda 1.0\ndef Xform "World" {}\n')
        archive.writestr("nested/payload.usdc", (tmp_path / "payload.usdc").read_bytes())
    result = scan(tmp_path, "a.usdz", container.getvalue())
    assert not result.errors
    assert any(
        row["member"] == "nested/payload.usdc"
        and row["location"] == "/World.metadata:default#json/nested"
        for row in result.hits
    )
    assert any(row["location"] == "/:documentation" for row in result.hits)
    assert any(row["location"].startswith("SdfExport:") for row in result.hits)


def test_data_identities_are_allowed(tmp_path):
    result = scan(tmp_path, "jenga01.json", b'{"jenga:mass": 1, "asset": "jenga01.usda"}')
    assert result.report()["passed"]


def test_invalid_layer_is_an_error(tmp_path):
    result = scan(tmp_path, "broken.usdc", b"not a layer")
    assert result.errors and not result.report()["passed"]


def test_numeric_npy_bytes_are_not_metadata_but_trailing_bytes_are(tmp_path):
    import numpy as np

    output = io.BytesIO()
    np.save(output, np.frombuffer(PRIVATE.encode(), dtype=np.uint8), allow_pickle=False)
    result = scan(tmp_path, "data.npy", output.getvalue())
    assert result.report()["passed"] and result.numeric_candidates
    changed = scan(tmp_path, "changed.npy", output.getvalue() + b"\0" + PRIVATE.encode())
    assert not changed.errors and any(
        row["location"].startswith("strings:") for row in changed.hits
    )


def test_hdf5_numeric_ranges_do_not_hide_attributes(tmp_path):
    import h5py
    import numpy as np

    output = io.BytesIO()
    with h5py.File(output, "w") as handle:
        handle.create_dataset("numeric", data=np.frombuffer(PRIVATE.encode(), dtype=np.uint8))
        handle.attrs["source"] = PRIVATE
    result = scan(tmp_path, "data.hdf5", output.getvalue())
    assert not result.errors
    assert any(row["location"] == "hdf5://@source" for row in result.hits)
    assert result.numeric_candidates


def test_all_patterns_and_image_trailer_are_inspected(tmp_path):
    from PIL import Image

    values = [
        PRIVATE,
        "C:" + "\\private\\operator",
        "operator@" + "example.org",
        "10." + "3.4.5",
        "fd00:" + ":1234",
        "aliy" + "un",
        "jenga" + "-sim",
        "N" + "AS",
        "tools/" + "v3",
        "redo_" + "claude",
        "mz" + "r",
    ]
    for index, value in enumerate(values):
        result = scan(tmp_path, f"sample-{index}.json", json.dumps({"metadata": value}).encode())
        assert result.hits and not result.errors, value
    picture = io.BytesIO()
    Image.new("RGB", (2, 2)).save(picture, format="PNG")
    assert scan(tmp_path, "trailer.png", picture.getvalue() + PRIVATE.encode()).hits


def test_public_versions_are_not_ip_addresses(tmp_path):
    result = scan(tmp_path, "version.json", b'{"isaac_sim": "6.1.0.0"}')
    assert result.report()["passed"]
    assert scan(tmp_path, "host.json", b'{"host": "6.1.' + b'0.0"}').hits
    assert scan(tmp_path, "slice.py", b"qpos[::15]").report()["passed"]
    assert scan(tmp_path, "endpoint.json", b'{"host": "http://[' + b":" + b':15]/"}').hits


def test_compressed_string_arrays_are_fully_decoded_without_an_extension(tmp_path):
    import h5py
    import numpy as np

    data = np.full(1200, b"public", dtype="S32")
    data[600] = PRIVATE.encode()
    output = io.BytesIO()
    with h5py.File(output, "w") as handle:
        handle.create_dataset("labels", data=data, compression="gzip")
        handle.attrs["labels"] = data
    result = scan(tmp_path, "opaque", output.getvalue())
    assert not result.errors
    assert any(row["location"] == "hdf5:labels/600" for row in result.hits)
    assert any(row["location"] == "hdf5://@labels/600" for row in result.hits)


def test_compressed_image_metadata_is_found_without_an_extension(tmp_path):
    from PIL import Image, PngImagePlugin

    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("source", PRIVATE, zip=True)
    output = io.BytesIO()
    Image.new("RGB", (2, 2)).save(output, format="PNG", pnginfo=metadata)
    result = scan(tmp_path, "opaque", output.getvalue())
    assert not result.errors and result.hits


def _compressed_path_png():
    from PIL import Image, PngImagePlugin
    from tower_sim.bench.public_surface import check_public

    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("source", "/" + "home/x/y", zip=True)
    output = io.BytesIO()
    Image.new("RGB", (2, 2)).save(output, format="PNG", pnginfo=metadata)
    payload = output.getvalue()
    assert b"zTXt" in payload
    check_public("payload.png", payload)  # The older byte/line gate cannot see this leak.
    return payload


def test_tree_gate_scanner_decodes_compressed_png(tmp_path):
    import subprocess

    from scripts.maintainer.check_public_tree import check_tree

    (tmp_path / "payload.png").write_bytes(_compressed_path_png())
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "payload.png"], cwd=tmp_path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=" + "test" + "@" + "example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=tmp_path,
        check=True,
    )
    with pytest.raises(ValueError, match="Public asset scan failed"):
        check_tree(tmp_path)


def test_wheel_gate_scanner_decodes_compressed_png(tmp_path):
    import subprocess

    from scripts.maintainer.check_public_tree import main

    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    wheel = tmp_path / "fixture.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("fixture/payload.png", _compressed_path_png())
    with pytest.raises(ValueError, match="Public wheel scan failed"):
        main(["--wheel", str(wheel), "--root", str(root)])


@pytest.mark.parametrize(
    "token",
    ["mZ" + "R", "Marco" + "zzxx" + "810", "Marco" + "wing" + "wing"],
    ids=["token-a", "token-b", "token-c"],
)
def test_personal_token_digests_preserve_case_and_word_boundaries(tmp_path, token):
    result = scan(tmp_path, "metadata.json", json.dumps({"owner": f"({token})"}).encode())
    assert not result.errors
    assert any(
        row["kind"] == "personal-username" and row["match"] == token and row["location"] == "/owner"
        for row in result.hits
    )
    benign = scan(tmp_path, "word.json", json.dumps({"owner": f"prefix{token}suffix"}).encode())
    assert benign.report()["passed"]
