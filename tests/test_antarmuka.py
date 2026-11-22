"""Pengujian interaksi Streamlit tanpa browser."""
import unittest
from pathlib import Path
from streamlit.testing.v1 import AppTest


class TestAntarmuka(unittest.TestCase):
    def test_training_and_test_sample_inspection(self):
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=90).run()
        self.assertEqual(len(app.exception), 0)
        app.slider(key="tab3_epochs").set_value(10)
        app.button(key="tab3_train_btn").click().run()
        self.assertEqual(len(app.exception), 0)
        model = app.session_state["model"]
        self.assertEqual(model.__class__.__module__, "src.anfis")
        self.assertEqual(len(model.loss_history), 10)
        app.radio(key="tab4_source").set_value("Sampel data uji").run()
        self.assertEqual(len(app.exception), 0)
        app.selectbox(key="tab4_rule").set_value(31).run()
        self.assertEqual(len(app.exception), 0)
        self.assertIn("f_R32", app.code[0].value)

    def test_missing_dataset(self):
        app = AppTest.from_string(
            "import streamlit as st\n"
            "from tabs.tab3_training import render_tab3_training\n"
            "from tabs.tab4_simulasi import render_tab4_simulasi\n"
            "st.session_state.data_pack = None\n"
            "st.session_state.model = None\n"
            "render_tab3_training()\nrender_tab4_simulasi()\n"
        ).run(timeout=90)
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.warning), 2)


if __name__ == "__main__":
    unittest.main()
