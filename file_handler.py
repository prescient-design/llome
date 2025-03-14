import s3fs
import os


class LocalOrS3Client:
    def __init__(self, init_s3: bool = False, **s3fs_kwargs):
        if init_s3:
            self.fs = s3fs.S3FileSystem(**s3fs_kwargs)

    def exists(self, path, **kwargs):
        if path.startswith("s3://"):
            return self.fs.exists(path, **kwargs)
        else:
            return os.path.exists(path)

    def ls(self, path, **kwargs):
        if path.startswith("s3://"):
            return self.fs.ls(path, **kwargs)
        else:
            return os.listdir(path)

    def get(self, rpath, lpath, **kwargs):
        """S3-only command!"""
        assert rpath.startswith("s3://"), f"rpath must start with s3://!"
        assert not lpath.startswith("s3://"), f"lpath cannot be a S3 path."
        return self.fs.get(rpath, lpath, **kwargs)

    def put(self, lpath, rpath, **kwargs):
        assert rpath.startswith("s3://"), f"rpath must start with s3://!"
        assert not lpath.startswith("s3://"), f"lpath cannot be a S3 path."
        """S3-only command!"""
        return self.fs.put(lpath, rpath, **kwargs)

    def open(self, fp, mode="rb", **kwargs):
        if fp.startswith("s3://"):
            return self.fs.open(fp, mode=mode, **kwargs)
        else:
            return open(fp, mode=mode, **kwargs)
