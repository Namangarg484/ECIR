"""Non-training regression checks for submission evidence/page-limit guards."""

import copy
from pathlib import Path
import tempfile
import unittest

from scripts.build_ecir_anonymous_artifact import audit

from scripts.audit_ecir_submission import (
    AuditFailure, DATASETS, RADIUS_V9, SUPPORT, TEX, read_csv,
    references_start_page, verify_secondary_bounds, verify_support_table,
)


class ReviewAnonymityTests(unittest.TestCase):
    def test_enabled_acknowledgements_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "ECIR.tex").write_text(
                r"\includeacknowledgementstrue", encoding="utf-8")
            self.assertTrue(any("acknowledgements enabled" in finding
                                for finding in audit(root)))

    def test_disabled_acknowledgements_and_commented_instruction_allowed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "ECIR.tex").write_text(
                r"\includeacknowledgementsfalse" + "\n" +
                r"% For publication: \includeacknowledgementstrue", encoding="utf-8")
            self.assertEqual(audit(root), [])


class PageLimitTests(unittest.TestCase):
    def test_twelve_content_pages_allowed(self):
        auxiliary = r"\newlabel{page:references-start}{{9}{13}{}{section*.20}{}}"
        self.assertEqual(references_start_page(auxiliary), 13)

    def test_reported_v9_length_rejected(self):
        auxiliary = r"\newlabel{page:references-start}{{9}{15}{}{section*.20}{}}"
        with self.assertRaises(AuditFailure):
            references_start_page(auxiliary)

    def test_missing_and_duplicate_marker_rejected(self):
        marker = r"\newlabel{page:references-start}{{9}{12}{}{section*.20}{}}"
        for auxiliary in ("", marker + "\n" + marker):
            with self.subTest(auxiliary=auxiliary), self.assertRaises(AuditFailure):
                references_start_page(auxiliary)


class ManuscriptEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Only small retained CSVs and manuscript text; no model imports or runs.
        source = TEX if TEX.is_file() else TEX.parent / "paper" / "ECIR.tex"
        cls.tex = source.read_text(encoding="utf-8")
        cls.coverage = read_csv(SUPPORT / "target_coverage.csv")
        cls.conditional = read_csv(SUPPORT / "warm_cold_metrics.csv")
        cls.secondary = {dataset: read_csv(RADIUS_V9 / dataset / "metrics.csv")
                         for dataset in DATASETS}

    def test_current_support_and_secondary_claims(self):
        verify_support_table(self.tex, self.coverage, self.conditional)
        verify_secondary_bounds(self.tex, self.secondary)

    def test_altered_cold_metric_rejected(self):
        altered = copy.deepcopy(self.conditional)
        target = next(row for row in altered if row["dataset"] == "amazon_music"
                      and row["method"] == "vsknn" and row["partition"] == "cold")
        target["NDCG@10"] = "0.01"
        with self.assertRaises(AuditFailure):
            verify_support_table(self.tex, self.coverage, altered)

    def test_partition_metric_must_reconcile(self):
        altered = copy.deepcopy(self.conditional)
        target = next(row for row in altered if row["dataset"] == "talkplay"
                      and row["method"] == "uniform" and row["partition"] == "warm")
        target["Recall@10"] = "0.9"
        with self.assertRaises(AuditFailure):
            verify_support_table(self.tex, self.coverage, altered)

    def test_large_secondary_effect_rejected(self):
        altered = copy.deepcopy(self.secondary)
        target = next(row for row in altered["talkplay"]
                      if row["method"] == "adaptive_smooth")
        target["MRR@50 mean"] = "0.5"
        with self.assertRaises(AuditFailure):
            verify_secondary_bounds(self.tex, altered)


if __name__ == "__main__":
    unittest.main()
