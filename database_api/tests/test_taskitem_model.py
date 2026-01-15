import unittest

from flask import Flask

from database_api.planexe_db_singleton import db
from database_api.model_taskitem import TaskItem, TaskState


class TestTaskItemModel(unittest.TestCase):
    def setUp(self):
        self.app = Flask(__name__)
        self.app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///:memory:"
        self.app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
        db.init_app(self.app)
        with self.app.app_context():
            db.create_all()

    def tearDown(self):
        with self.app.app_context():
            db.session.remove()
            db.drop_all()

    def test_stop_request_fields_default(self):
        with self.app.app_context():
            task = TaskItem(
                state=TaskState.pending,
                prompt="Test prompt",
                user_id="test_user",
            )
            db.session.add(task)
            db.session.commit()

            fetched = db.session.get(TaskItem, task.id)
            self.assertIsNotNone(fetched)
            self.assertTrue(hasattr(fetched, "stop_requested"))
            self.assertTrue(hasattr(fetched, "stop_requested_timestamp"))
            self.assertFalse(bool(fetched.stop_requested))
