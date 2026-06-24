import tempfile
import unittest
from pathlib import Path

from firecls.data.dataset import resolve_indexed_image_path


class DatasetPathTests(unittest.TestCase):
    def test_rebases_windows_path_beneath_images_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nested = root / "sequence" / "frame.jpg"
            nested.parent.mkdir()
            nested.touch()
            raw = r"E:\project\datasets\FASDD_CV\images\sequence\frame.jpg"
            self.assertEqual(resolve_indexed_image_path(raw, root), nested)


if __name__ == "__main__":
    unittest.main()
