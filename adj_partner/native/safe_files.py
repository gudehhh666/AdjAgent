"""Bounded regular-file snapshots for trusted native adapters on POSIX.

Open every path component by descriptor with no-follow; never read FIFOs,
devices, hardlink aliases or Store-private paths. This is conservative input
validation, not an OS sandbox against arbitrary hostile local Python processes.
"""

from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
import os
from pathlib import Path
import stat

from ..tools.policy import _credential_path


@dataclass(frozen=True)
class InputFile:
    path: str
    content: bytes = field(repr=False)
    sha256: str
    size: int

    def manifest(self):
        return {"path": self.path, "sha256": self.sha256, "size": self.size}


def _checked_path(context, path):
    context.check_cancelled()
    candidate = Path(path).expanduser().absolute()
    if ".." in candidate.parts or _credential_path(candidate):
        raise PermissionError("Traversal and credential input paths are denied")
    if candidate.is_relative_to(context.store.root.resolve()):
        raise PermissionError("Store-private files are not native input files")
    resolved = context.check_path(candidate)
    if resolved != candidate or resolved.is_relative_to(context.store.root.resolve()):
        raise PermissionError("Symlink components and Store-private inputs are denied")
    return candidate


@contextmanager
def open_regular_input(context, path):
    """Yield a regular input FD while retaining its checked parent descriptor."""
    candidate = _checked_path(context, path)
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
    parent = os.open(candidate.anchor, flags | os.O_DIRECTORY)
    file_fd = None
    try:
        for part in candidate.parts[1:-1]:
            child = os.open(part, flags | os.O_DIRECTORY, dir_fd=parent)
            os.close(parent)
            parent = child
        file_fd = os.open(candidate.name or ".", flags, dir_fd=parent)
        before = os.fstat(file_fd)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("Input must be a regular file, not a directory, device or pipe")
        if before.st_nlink != 1:
            raise PermissionError("Hardlinked input files are denied to prevent private-file aliases")
        yield candidate, file_fd, before
        after = os.fstat(file_fd)
        if _signature(before) != _signature(after):
            raise ValueError("Input file changed during read; retry with stable input")
        current = os.stat(candidate, follow_symlinks=False)
        if _signature(current) != _signature(after) or candidate.resolve(strict=True) != candidate:
            raise PermissionError("Input path changed during read")
        parent_now = os.stat(candidate.parent, follow_symlinks=False)
        parent_open = os.fstat(parent)
        if (parent_now.st_dev, parent_now.st_ino) != (parent_open.st_dev, parent_open.st_ino):
            raise PermissionError("Input parent changed during read")
        _checked_path(context, candidate)
    finally:
        if file_fd is not None:
            os.close(file_fd)
        os.close(parent)


def _signature(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_nlink,
            info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def read_input(context, path, *, maximum_bytes=4 * 1024 * 1024):
    if type(maximum_bytes) is not int or not 1 <= maximum_bytes <= 128 * 1024 * 1024:
        raise ValueError("Input byte limit must be an integer in 1..128 MiB")
    with open_regular_input(context, path) as (candidate, fd, info):
        if info.st_size > maximum_bytes:
            raise ValueError("Input exceeds byte limit")
        chunks, length = [], 0
        while True:
            context.check_cancelled()
            chunk = os.read(fd, min(65536, maximum_bytes + 1 - length))
            if not chunk:
                break
            chunks.append(chunk)
            length += len(chunk)
            if length > maximum_bytes:
                raise ValueError("Input exceeds byte limit while reading")
        content = b"".join(chunks)
    return InputFile(str(candidate), content, hashlib.sha256(content).hexdigest(), len(content))


def list_inputs(context, directory, *, suffix, maximum_files=64):
    """Bounded descriptor-based discovery; each later file read is rechecked."""
    if (type(maximum_files) is not int or not 1 <= maximum_files <= 128
            or not isinstance(suffix, str) or not suffix.startswith(".") or len(suffix) > 32 or "/" in suffix):
        raise ValueError("Invalid bounded file-discovery options")
    candidate = _checked_path(context, directory)
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC | os.O_DIRECTORY
    fd = os.open(candidate.anchor, flags)
    try:
        for part in candidate.parts[1:]:
            child = os.open(part, flags, dir_fd=fd)
            os.close(fd)
            fd = child
        before = os.fstat(fd)
        results = []
        with os.scandir(fd) as entries:
            for scanned, entry in enumerate(entries, 1):
                context.check_cancelled()
                if scanned > 512:
                    raise ValueError("Input directory exceeds the 512-entry discovery limit")
                if not entry.name.endswith(suffix):
                    continue
                info = entry.stat(follow_symlinks=False)
                if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                    raise PermissionError("Dataset entries must be regular non-linked files")
                results.append(str(candidate / entry.name))
                if len(results) > maximum_files:
                    raise ValueError("Input directory exceeds requested file limit")
        if _signature(before) != _signature(os.fstat(fd)) or _signature(before) != _signature(os.stat(candidate, follow_symlinks=False)):
            raise PermissionError("Input directory changed during discovery")
        _checked_path(context, candidate)
        return sorted(results)
    finally:
        os.close(fd)
