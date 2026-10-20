"""CPU checks for paths, missing model shards and launcher mode isolation."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).parents[1]
ENTRY = ROOT/'scripts/3d/train/train_scanqa_foresight.py'
spec = importlib.util.spec_from_file_location('scanqa_launcher', ENTRY)
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


class LauncherTests(unittest.TestCase):
    def test_relative_and_absolute_roots(self):
        config = {'data': {'root': 'data/scanqa_foresight'},
                  'training': {'output_root': '/tmp/scanqa-test-output'}}
        resolved = entry.resolve_config_paths(copy.deepcopy(config))
        self.assertEqual(Path(resolved['data']['root']), ROOT/'data/scanqa_foresight')
        self.assertEqual(resolved['training']['output_root'], '/tmp/scanqa-test-output')

    def test_missing_and_present_model_shards(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path/'model.safetensors.index.json').write_text(json.dumps(
                {'weight_map': {'a':'weights-1.safetensors', 'b':'weights-2.safetensors'}}))
            (path/'weights-1.safetensors').write_bytes(b'present')
            with self.assertRaises(FileNotFoundError):
                entry.check_weight_files(path)
            (path/'weights-2.safetensors').write_bytes(b'present')
            entry.check_weight_files(path)
            (path/'weights-2.safetensors').write_bytes(b'')
            with self.assertRaises(FileNotFoundError):
                entry.check_weight_files(path)

    def test_check_mode_cannot_start_training(self):
        for flag in ['--check-data', '--check-loader']:
            result = subprocess.run([sys.executable, str(ENTRY), flag, '--execute', '--gpus', '0'],
                                    cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn('separately from --execute', result.stderr)

    def test_three_experiments_keep_matched_batch_and_input(self):
        import yaml
        config = yaml.safe_load((ROOT/'configs/foresight/scanqa_pilot.yaml').read_text())
        for name, experiment in config['experiments'].items():
            arguments = entry.training_arguments(config, experiment, Path('data.yaml'),
                Path('output'), 'qwen35-test', 2)
            parsed = dict(zip(arguments[::2], arguments[1::2]))
            self.assertEqual(parsed['--frames_upbound'], '32')
            self.assertEqual(parsed['--gradient_accumulation_steps'], '4')
            self.assertEqual(parsed['--foresight_video_loss_weight'], '0.1' if name=='E03' else '0.0')


if __name__ == '__main__':
    unittest.main(verbosity=2)
