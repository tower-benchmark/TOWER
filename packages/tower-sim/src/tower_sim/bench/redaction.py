"""Exact configuration equality with a fail-closed JSON pointer diagnostic."""


def matches_exact(original, packaged):
    """Verify exact values, without accepting provenance tokens or wildcards.

    Return True on success; raise ValueError at the first mismatching JSON
    pointer. The redaction marker is an ordinary literal string.
    """

    def child(pointer, key):
        return pointer + "/" + str(key).replace("~", "~0").replace("/", "~1")

    def mismatch(pointer):
        raise ValueError(f"Configuration mismatch at JSON pointer {pointer!r}")

    def compare(left, right, pointer):
        if isinstance(right, dict):
            if not isinstance(left, dict):
                mismatch(pointer)
            if left.keys() != right.keys():
                mismatch(child(pointer, sorted(left.keys() ^ right.keys())[0]))
            for key in right:
                compare(left[key], right[key], child(pointer, key))
        elif isinstance(right, list):
            if not isinstance(left, list):
                mismatch(pointer)
            if len(left) != len(right):
                mismatch(child(pointer, min(len(left), len(right))))
            for index, (a, b) in enumerate(zip(left, right)):
                compare(a, b, child(pointer, index))
        elif not left == right:  # noqa: SIM201 -- preserve the original equality operation
            mismatch(pointer)

    compare(original, packaged, "")
    return True
