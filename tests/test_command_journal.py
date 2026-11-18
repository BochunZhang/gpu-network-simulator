import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import experiment_handlers as handlers


class CommandJournalTest(unittest.TestCase):
    def test_records_argv_and_failed_command_without_environment(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            handlers.command([sys.executable, '-c', 'print("hello world")'], output / 'ok.log')
            with self.assertRaisesRegex(RuntimeError, 'returned 7'):
                handlers.command([sys.executable, '-c', 'raise SystemExit(7)'], output / 'failed.log')
            journal = json.loads((output / 'commands.json').read_text())
            self.assertEqual(len(journal), 2)
            self.assertEqual(journal[0]['argv'][2], 'print("hello world")')
            self.assertEqual(journal[0]['state'], 'succeeded')
            self.assertEqual(journal[1]['returncode'], 7)
            self.assertEqual(journal[1]['state'], 'failed')
            self.assertNotIn('env', journal[0])
            self.assertIn('finished_at', journal[1])
            self.assertEqual((output / 'ok.log').read_text().strip(), 'hello world')
