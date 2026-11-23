"""Process locks for immutable artifact publication on POSIX and Windows."""
import os


def lock_file(stream, *, blocking=True):
    stream.flush()
    if os.name == 'nt':
        import msvcrt
        if os.fstat(stream.fileno()).st_size == 0:
            stream.write('\0')
            stream.flush()
        os.lseek(stream.fileno(), 0, os.SEEK_SET)
        msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK if blocking else msvcrt.LK_NBLCK, 1)
    else:
        import fcntl
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))


def unlock_file(stream):
    if os.name == 'nt':
        import msvcrt
        os.lseek(stream.fileno(), 0, os.SEEK_SET)
        msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
