import tempfile
import unittest
from pathlib import Path

from roboatom.annotation.app import AnnotationStore


class LocalStreamCacheBustingTest(unittest.TestCase):
    def test_local_stream_url_changes_with_file_version(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            video_path = Path(tmp_dir) / "video.mp4"
            video_path.write_bytes(b"first")
            first = AnnotationStore._build_stream_url(None, 3, "head", video_path)
            video_path.write_bytes(b"second-version")
            second = AnnotationStore._build_stream_url(None, 3, "head", video_path)

        self.assertTrue(first.startswith("/api/stream/3/head?v="))
        self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
