import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import classifier_config as config


class ClassifierConfigTests(unittest.TestCase):
    def test_registry_and_explicit_candidate(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);registry=root/'maps.json'
            (root/'active.pt').write_bytes(b'active');candidate=root/'candidate.pt';candidate.write_bytes(b'candidate')
            registry.write_text(json.dumps(dict(active_classifier='active.pt')))
            with patch.object(config,'ROOT',root),patch.object(config,'REGISTRY',registry):
                self.assertEqual(config.classifier_path(),root/'active.pt')
                self.assertEqual(config.classifier_path(candidate),candidate)
                with self.assertRaises(FileNotFoundError):config.classifier_path('missing.pt')
                self.assertEqual(json.loads(registry.read_text())['active_classifier'],'active.pt')


if __name__=='__main__':unittest.main()
