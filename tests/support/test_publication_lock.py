"""An immutable output has one writer even on Windows."""
from pathlib import Path
import tempfile
from unittest import TestCase
from src.support.storage.locking import lock_file, unlock_file


class PublicationLock(TestCase):
    def test_second_writer_cannot_acquire_locked_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'artifact.lock'
            with path.open('a+',encoding='utf-8') as first, path.open('a+',encoding='utf-8') as second:
                lock_file(first,blocking=False)
                with self.assertRaises(OSError):lock_file(second,blocking=False)
                unlock_file(first)
                lock_file(second,blocking=False)
                unlock_file(second)
