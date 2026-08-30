import json
import os
import tempfile
import unittest
from audiobook_factory.progress_io import read_progress_file, write_progress_file

class TestProgressJsonUtils(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_valid_json_load(self):
        file_path = os.path.join(self.temp_dir.name, "valid.json")
        data = {
            "book_title": "Test Book",
            "book_path": "",
            "voice_file": "",
            "settings": {"author": "Tester"},
            "chapters": [{"num": 1, "title": "Chapter 1", "status": "completed"}]
        }
        write_progress_file(file_path, data)

        res = read_progress_file(file_path)
        self.assertEqual(res["book_title"], "Test Book")

    def test_utf8_bom_json_load(self):
        file_path = os.path.join(self.temp_dir.name, "bom.json")
        data = {
            "book_title": "BOM Book",
            "settings": {},
            "chapters": []
        }
        with open(file_path, "w", encoding="utf-8-sig") as f:
            json.dump(data, f)

        res = read_progress_file(file_path)
        self.assertEqual(res["book_title"], "BOM Book")

    def test_corrupted_json_file_raises(self):
        file_path = os.path.join(self.temp_dir.name, "corrupted.json")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("corrupted content string")

        with self.assertRaises(ValueError):
            read_progress_file(file_path)

    def test_leading_garbage_auto_healing(self):
        file_path = os.path.join(self.temp_dir.name, "garbage.json")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write('chp{\n "book_title": "Auto Healed Book",\n "settings": {},\n "chapters": []\n}')

        data = read_progress_file(file_path)
        self.assertEqual(data["book_title"], "Auto Healed Book")

    def test_audiobook_config_kwargs(self):
        from audiobook_factory.pipeline import AudiobookConfig
        cfg = AudiobookConfig(
            selected_chapters=["Chapter 1"],
            regen_missing=True,
        )
        self.assertEqual(cfg.selected_chapters, ["Chapter 1"])
        self.assertTrue(cfg.regen_missing)

        cfg_from_dict = AudiobookConfig.from_dict({
            "selected_chapters": ["Chapter 2"],
            "unknown_extra_arg": "ignored_value"
        })
        self.assertEqual(cfg_from_dict.selected_chapters, ["Chapter 2"])

if __name__ == "__main__":
    unittest.main()
