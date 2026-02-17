import os
import shutil
import time
import unittest
import uuid

from worker_plan.worker_plan_internal.utils.purge_old_runs import purge_old_runs


class TestPurgeOldRuns(unittest.TestCase):
    def setUp(self):
        self.test_run_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "test_run"))
        if os.path.exists(self.test_run_dir):
            shutil.rmtree(self.test_run_dir)
        os.makedirs(self.test_run_dir, exist_ok=True)

        self.uuid_old_valid = str(uuid.uuid4())
        self.uuid_recent_valid = str(uuid.uuid4())
        self.uuid_old_missing_start = str(uuid.uuid4())
        self.uuid_old_missing_plan = str(uuid.uuid4())
        self.uuid_old_zip = str(uuid.uuid4()) + ".zip"

        self._create_run_dir(self.uuid_old_valid, hours_old=2.0, with_start=True, with_plan=True)
        self._create_run_dir(self.uuid_recent_valid, hours_old=0.1, with_start=True, with_plan=True)
        self._create_run_dir(self.uuid_old_missing_start, hours_old=2.0, with_start=False, with_plan=True)
        self._create_run_dir(self.uuid_old_missing_plan, hours_old=2.0, with_start=True, with_plan=False)
        self.legacy_old_prefixed = "PlanExe_19841231_195936"

        self._create_run_dir("not-a-uuid", hours_old=2.0, with_start=True, with_plan=True)
        self._create_run_dir(self.legacy_old_prefixed, hours_old=2.0, with_start=True, with_plan=True)
        self._create_file(self.uuid_old_zip, hours_old=2.0)
        self._create_file("not-a-uuid.zip", hours_old=2.0)
        self._create_file("random.txt", hours_old=2.0)

    def tearDown(self):
        if os.path.exists(self.test_run_dir):
            shutil.rmtree(self.test_run_dir)

    def _set_mtime(self, path: str, hours_old: float):
        mtime = time.time() - (hours_old * 3600)
        os.utime(path, (mtime, mtime))

    def _create_run_dir(self, dirname: str, hours_old: float, with_start: bool, with_plan: bool):
        path = os.path.join(self.test_run_dir, dirname)
        os.makedirs(path, exist_ok=True)
        if with_start:
            with open(os.path.join(path, "001-1-start_time.json"), "w", encoding="utf-8") as f:
                f.write("{}")
        if with_plan:
            with open(os.path.join(path, "001-2-plan.txt"), "w", encoding="utf-8") as f:
                f.write("plan")
        self._set_mtime(path, hours_old)

    def _create_file(self, filename: str, hours_old: float):
        path = os.path.join(self.test_run_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write("dummy")
        self._set_mtime(path, hours_old)

    def test_purge_uuid_run_dirs_with_required_files_only(self):
        purge_old_runs(self.test_run_dir, max_age_hours=1.0, prefix="")

        self.assertFalse(os.path.exists(os.path.join(self.test_run_dir, self.uuid_old_valid)))
        self.assertTrue(os.path.exists(os.path.join(self.test_run_dir, self.uuid_recent_valid)))
        self.assertTrue(os.path.exists(os.path.join(self.test_run_dir, self.uuid_old_missing_start)))
        self.assertTrue(os.path.exists(os.path.join(self.test_run_dir, self.uuid_old_missing_plan)))
        self.assertTrue(os.path.exists(os.path.join(self.test_run_dir, "not-a-uuid")))
        self.assertTrue(os.path.exists(os.path.join(self.test_run_dir, self.legacy_old_prefixed)))
        self.assertTrue(os.path.exists(os.path.join(self.test_run_dir, self.uuid_old_zip)))
        self.assertTrue(os.path.exists(os.path.join(self.test_run_dir, "not-a-uuid.zip")))
        self.assertTrue(os.path.exists(os.path.join(self.test_run_dir, "random.txt")))

    def test_purge_can_target_legacy_prefixed_runs_when_prefix_is_explicit(self):
        purge_old_runs(self.test_run_dir, max_age_hours=1.0, prefix="PlanExe_")

        self.assertFalse(os.path.exists(os.path.join(self.test_run_dir, self.legacy_old_prefixed)))
        self.assertFalse(os.path.exists(os.path.join(self.test_run_dir, self.uuid_old_valid)))
        self.assertTrue(os.path.exists(os.path.join(self.test_run_dir, self.uuid_recent_valid)))
        self.assertTrue(os.path.exists(os.path.join(self.test_run_dir, "not-a-uuid")))


if __name__ == "__main__":
    unittest.main()
