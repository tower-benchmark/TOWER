"""Atomic, hash-verified public downloads with bounded transport retries."""

import hashlib
import logging
import os
import ssl
import tempfile
import time
from http.client import IncompleteRead
from pathlib import Path
from urllib.error import HTTPError, URLError

from .resources import sha

RETRY_ERRORS = (URLError, ConnectionError, TimeoutError, ssl.SSLError, IncompleteRead)
RETRY_EVENTS: list[dict] = []
ATTEMPTS = 5


def download_file(url, target, expected, *, size=None, opener):
    """Resume by verified file; restart a failed stream, never publish partial bytes."""
    target = Path(target)
    if (
        target.is_file()
        and (size is None or target.stat().st_size == size)
        and sha(target) == expected
    ):
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, ATTEMPTS + 1):
        fd, temporary = tempfile.mkstemp(prefix=".download-", dir=target.parent)
        try:
            digest, count = hashlib.sha256(), 0
            with os.fdopen(fd, "wb") as stream, opener(url, timeout=120) as response:
                expected_length = getattr(response, "length", None)
                while chunk := response.read(1024 * 1024):
                    stream.write(chunk)
                    digest.update(chunk)
                    count += len(chunk)
                if expected_length is not None and count < expected_length:
                    raise IncompleteRead(b"", expected_length - count)
            if digest.hexdigest() != expected or (size is not None and count != size):
                raise ValueError(f"Downloaded checksum/size differs: {target.name}")
            Path(temporary).replace(target)
            return
        except RETRY_ERRORS as error:
            if isinstance(error, HTTPError) and 400 <= error.code < 500:
                raise
            if attempt == ATTEMPTS:
                raise
            RETRY_EVENTS.append(dict(url=url, attempt=attempt, error_type=type(error).__name__))
            logging.getLogger(__name__).warning("Retrying public download %s/5: %s", attempt, url)
            time.sleep(min(2 ** (attempt - 1), 8))
        finally:
            Path(temporary).unlink(missing_ok=True)
