from PyQt5.QtWidgets import QMessageBox, QDialog, QVBoxLayout, QLabel, QProgressBar, QApplication, QPushButton, QFileDialog, QHBoxLayout
import json


def print_error(s, extra_info="", severity=QMessageBox.Critical):
        dlg = QMessageBox()
        dlg.setWindowTitle("Error Message")
        if isinstance(s, tuple):
            text = f"For path {s[1]} the following error occurred:\n {s[0]}"
        else:
            text = s
        if extra_info != "":
            text += f"\n{extra_info}"
        dlg.setText(text)
        dlg.setIcon(severity)
        dlg.exec()


class ProgressWindow(QDialog):
    def __init__(self, total_steps, text="Shuffling", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Progress")
        self.setGeometry(600, 300, 300, 100)
        self.text = text

        layout = QVBoxLayout(self)

        self.label = QLabel("Progress:", self)
        layout.addWidget(self.label)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setMaximum(total_steps)
        layout.addWidget(self.progress_bar)

        self.setFixedSize(400, self.sizeHint().height())

    def update_progress(self, step):
        self.progress_bar.setValue(step)
        self.label.setText(f"{self.text} {step}/{self.progress_bar.maximum()}...")
        QApplication.processEvents()


class SaveSessionSettingsDialog(QDialog):
    """Dialog to choose a path and (eventually) save current session settings to JSON.

    For now, saving is a placeholder; this centralizes the UI so other widgets can reuse it.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Save Current Session Settings")
        self.setMinimumWidth(500)
        self.setMinimumHeight(200)
        self.selected_path = None

        layout = QVBoxLayout(self)

        title_label = QLabel("Save Current Parameters to JSON")
        title_label.setStyleSheet("font-size: 14px; font-weight: bold; margin: 10px;")
        layout.addWidget(title_label)

        desc_label = QLabel(
            "This will save all current visualization and analysis parameters including:\n"
            "• 3D Visualization settings\n"
            "• Advanced 3D Visualization settings\n"
            "• Event-based Shuffling settings\n\n"
            "Choose a location to save. Saving will happen immediately and this window will close."
        )
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("margin: 10px; padding: 10px; background-color: #f0f0f0;")
        layout.addWidget(desc_label)

        # Buttons row
        button_row = QHBoxLayout()

        browse_button = QPushButton("Browse…")
        browse_button.clicked.connect(self._on_browse)
        button_row.addWidget(browse_button)

        button_row.addStretch()

        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_row.addWidget(cancel_button)

        layout.addLayout(button_row)

    def _on_browse(self):
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Session Settings",
            "session_settings.json",
            "JSON Files (*.json)"
        )
        if path:
            if not path.lower().endswith('.json'):
                path += '.json'
            self.selected_path = path
            # Immediately save and close after choosing the file
            self._on_save()

    def _on_save(self):
        if not self.selected_path:
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle("No File Selected")
            msg.setText("Please choose a destination file for the settings.")
            msg.exec_()
            return
        try:
            data = self._collect_settings()
            with open(self.selected_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)
            self.accept()
        except Exception as e:
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Critical)
            msg.setWindowTitle("Save Failed")
            msg.setText("Failed to save session settings.")
            msg.setInformativeText(str(e))
            msg.exec_()

    def _collect_settings(self):
        """Collect and return only the three requested categories with parameters
        actually passed to downstream functions:
        - 3D Visualization (args passed to base_visualization via change_func)
        - Advanced Visualization (args passed to VisualizationAdvancedWidget.set_data)
        - Event based Shuffling (args passed to event_based_shuffle_analysis)
        """
        owner = self.parent()

        def safe_get(attr, default=None):
            try:
                return getattr(owner, attr)
            except Exception:
                return default

        def safe_text(widget_name, default=None):
            w = safe_get(widget_name)
            try:
                return w.text() if w is not None else default
            except Exception:
                return default

        def safe_int(widget_name, default=None):
            txt = safe_text(widget_name, None)
            try:
                return int(txt) if txt is not None and txt != "" else default
            except Exception:
                return default

        def safe_current_text(widget_name, default=None):
            w = safe_get(widget_name)
            try:
                return w.currentText() if w is not None else default
            except Exception:
                return default

        def safe_slider_value(widget_name, default=None):
            w = safe_get(widget_name)
            try:
                return int(w.value()) if w is not None else default
            except Exception:
                return default

        def safe_checked(widget_name, default=None):
            w = safe_get(widget_name)
            try:
                return bool(w.isChecked()) if w is not None else default
            except Exception:
                return default

        settings = {
            "3D Visualization": {
                "Signal Settings": {},
                "Co-firing": {},
                "Shuffling": {},
            },
            "Advanced Visualization": {
                "FPR visualization": {},
                "FPR Shuffling": {},
            },
            "Event based Shuffling": {},
        }

        # 1) 3D Visualization (values passed to base_visualization via change_func)
        try:
            vis_func = safe_current_text("dropdown_3D_functions")
            data_type_ui = safe_current_text("dropdown_3D_data_types")
            scaling = safe_slider_value("slider_3D_scaling")
            cells_to_visualize = safe_current_text("cmb_3D_which_cells")
            smoothing_size = safe_int("input_smoothing_size")
            smoothing_type = (safe_current_text("dropdown_smoothing_type") or "").lower() or None
            window_size = safe_int("input_3D_window_size", 1)
            normalize = safe_checked("chkbox_3D_normalize")
            average = safe_checked("chkbox_3D_average")
            cumulative = safe_checked("chkbox_3D_cumulative")

            # Compute the actual data_type string exactly like visualize_3D
            resolved_data_type = data_type_ui
            if resolved_data_type == "Binary Transient":
                resolved_data_type = "E"
            if resolved_data_type in ["C", "DFF"] and vis_func == "Transient Visualization":
                resolved_data_type = f"{resolved_data_type}_cumulative" if cumulative else f"{resolved_data_type}_transient"

            # If not transient visualization, window_size should be 1
            window_size_resolved = window_size if vis_func == "Transient Visualization" else 1

            settings["3D Visualization"]["Signal Settings"] = {
                "function": vis_func,
                "data_type": resolved_data_type,
                "scaling": scaling,
                "cells_group": cells_to_visualize,
                "smoothing_size": smoothing_size,
                "smoothing_type": smoothing_type,
                "window_size": window_size_resolved,
                "normalize": normalize,
                "average": average,
                "cumulative": cumulative,
            }

            # Co-firing (visual overlay)
            cof_enabled = safe_checked("cofiring_chkbox")
            cof_window = safe_int("cofiring_window_size")
            cof_shareA = safe_checked("cofiring_shareA_chkbox")
            cof_shareB = safe_checked("cofiring_shareB_chkbox")
            cof_direction = (safe_current_text("cofiring_direction_dropdown") or "").lower() or None
            settings["3D Visualization"]["Co-firing"] = {
                "enabled": cof_enabled,
                "window_size": cof_window,
                "shareA": cof_shareA,
                "shareB": cof_shareB,
                "direction": cof_direction,
            }

            # Shuffling (cofiring shuffles in 3D section)
            shuf_group = safe_current_text("cmb_shuffle_which_cells")
            shuf_verified_only = safe_checked("chkbox_shuffle_verified_only")
            shuf_spatial = safe_checked("chkbox_shuffle_spatial")
            shuf_temporal = safe_checked("chkbox_shuffle_temporal")
            shuf_cof_win = safe_int("shuffling_cofiring_window_size")
            shuf_shareA = safe_checked("shuffling_temporal_shareA_chkbox")
            shuf_shareB = safe_checked("shuffling_temporal_shareB_chkbox")
            shuf_direction = (safe_current_text("shuffling_cofiring_direction_dropdown") or "").lower() or None
            shuf_num = safe_int("shuffle_num_shuffles")
            settings["3D Visualization"]["Shuffling"] = {
                "verified_only": shuf_verified_only,
                "spatial": shuf_spatial,
                "temporal": shuf_temporal,
                "cofiring_window_size": shuf_cof_win,
                "shareA": shuf_shareA,
                "shareB": shuf_shareB,
                "direction": shuf_direction,
                "num_shuffles": shuf_num,
            }
        except Exception:
            # Leave empty if controls not available
            pass

        # 2) Advanced Visualization (values passed to set_data and shuffling controls)
        try:
            adv_window_size = safe_int("input_3D_advanced_window_size")
            adv_readout = safe_current_text("dropdown_3D_advanced_readout")
            adv_fpr = safe_current_text("dropdown_3D_advanced_fpr")
            adv_scaling = safe_slider_value("slider_3D_advanced_scaling")
            adv_cells_group = safe_current_text("cmb_3D_advanced_which_cells")

            settings["Advanced Visualization"]["FPR visualization"] = {
                "window_size": adv_window_size,
                "readout": adv_readout,
                "fpr": adv_fpr,
                "scaling": adv_scaling,
            }

            # FPR Shuffling settings
            adv_shuf_spatial = safe_checked("visualization_3D_advanced_shuffle_spatial")
            adv_shuf_temporal = safe_checked("visualization_3D_advanced_shuffle_temporal")
            adv_shuf_num = safe_int("input_advanced_shuffling_num")
            # Anchor is always True in code; include for completeness
            settings["Advanced Visualization"]["FPR Shuffling"] = {
                "spatial": adv_shuf_spatial,
                "temporal": adv_shuf_temporal,
                "num_shuffles": adv_shuf_num,
                "anchor": True,
            }
        except Exception:
            pass

        # 3) Event based Shuffling (values passed to event_based_shuffle_analysis)
        try:
            e_event_type = safe_current_text("event_based_event_dropdown")
            e_window_size = safe_int("event_based_window_size_input")
            e_lag = safe_int("event_based_lag_input")
            e_num_subwindows = safe_slider_value("event_based_subwindows_slider")
            e_num_shuffles = safe_int("event_based_shuffles_input")
            e_amplitude_anchored = safe_checked("event_based_amplitude_anchored")

            # Determine shuffle type exactly as UI enforces
            shuffle_temporal = safe_checked("event_based_shuffle_temporal", False)
            shuffle_spatial = safe_checked("event_based_shuffle_spatial", False)
            if shuffle_spatial:
                e_shuffle_type = "spatial"
            else:
                e_shuffle_type = "temporal" if shuffle_temporal else None

            # Instead of saving explicit cell IDs (session-specific), save group selection
            which_cells = safe_current_text("cmb_event_based_which_cells")

            settings["Event based Shuffling"] = {
                "event_type": e_event_type,
                "window_size": e_window_size,
                "lag": e_lag,
                "num_subwindows": e_num_subwindows,
                "num_shuffles": e_num_shuffles,
                "amplitude_anchored": e_amplitude_anchored,
                "shuffle_type": e_shuffle_type,
                "which_cells": which_cells,
            }
        except Exception:
            pass

        return settings


class LoadSessionSettingsDialog(QDialog):
    """Dialog to choose a JSON file and apply session settings to the parent widget.

    Minimal UI: a Browse button triggers immediate load and close; errors are shown if any.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Load Session Settings")
        self.setMinimumWidth(500)
        self.setMinimumHeight(160)

        layout = QVBoxLayout(self)

        title_label = QLabel("Load Parameters from JSON")
        title_label.setStyleSheet("font-size: 14px; font-weight: bold; margin: 10px;")
        layout.addWidget(title_label)

        desc_label = QLabel(
            "Select a previously saved settings JSON. Values will be applied immediately "
            "(scalars or single-item lists are both accepted)."
        )
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("margin: 10px; padding: 10px; background-color: #f0f0f0;")
        layout.addWidget(desc_label)

        # Buttons row
        button_row = QHBoxLayout()

        browse_button = QPushButton("Browse…")
        browse_button.clicked.connect(self._on_browse)
        button_row.addWidget(browse_button)

        button_row.addStretch()

        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_row.addWidget(cancel_button)

        layout.addLayout(button_row)

    @staticmethod
    def _scalar(value):
        # Accept scalars or lists; if list, take first element
        if isinstance(value, list):
            return value[0] if value else None
        return value

    def _set_combo_text(self, widget, text):
        if widget is None or text is None:
            return
        # Try exact match first
        for i in range(widget.count()):
            if widget.itemText(i) == text:
                widget.setCurrentIndex(i)
                return
        # Fallback to case-insensitive match
        lower = text.lower()
        for i in range(widget.count()):
            if widget.itemText(i).lower() == lower:
                widget.setCurrentIndex(i)
                return

    def _apply_settings(self, data):
        owner = self.parent()
        if owner is None:
            return

        # Small helpers
        def set_line(name, value):
            w = getattr(owner, name, None)
            val = self._scalar(value)
            if w is not None and val is not None:
                w.setText(str(val))

        def set_check(name, value):
            w = getattr(owner, name, None)
            val = self._scalar(value)
            if w is not None and val is not None:
                w.setChecked(bool(val))

        def set_slider(name, value, vmin=None, vmax=None):
            w = getattr(owner, name, None)
            val = self._scalar(value)
            if w is not None and val is not None:
                try:
                    iv = int(val)
                    if vmin is not None:
                        iv = max(vmin, iv)
                    if vmax is not None:
                        iv = min(vmax, iv)
                    w.setValue(iv)
                except Exception:
                    pass

        def set_combo(name, value, mapping=None):
            w = getattr(owner, name, None)
            val = self._scalar(value)
            if val is None or w is None:
                return
            if mapping and val in mapping:
                val = mapping[val]
            self._set_combo_text(w, str(val))

        # 3D Visualization -> Signal Settings
        sig = (data or {}).get("3D Visualization", {}).get("Signal Settings", {})
        if sig:
            # function first (this changes available options downstream)
            set_combo("dropdown_3D_functions", sig.get("function"))

            # data_type resolution back into UI choices
            resolved = self._scalar(sig.get("data_type"))
            base_dtype = None
            if resolved in ("C", "DFF"):
                base_dtype = resolved
            elif isinstance(resolved, str) and (resolved.startswith("C_") or resolved.startswith("DFF_")):
                base_dtype = resolved.split("_")[0]
            elif resolved == "E":
                base_dtype = "Binary Transient"

            if base_dtype == "Binary Transient":
                set_combo("dropdown_3D_data_types", "Binary Transient")
            elif base_dtype in ("C", "DFF"):
                set_combo("dropdown_3D_data_types", base_dtype)

            # group and numerics
            set_combo("cmb_3D_which_cells", sig.get("cells_group"))
            set_slider("slider_3D_scaling", sig.get("scaling"), 1, 1000)
            # smoothing
            set_line("input_smoothing_size", sig.get("smoothing_size"))
            # Map smoothing type to UI label
            smoothing_map = {"mean": "Mean"}
            set_combo("dropdown_smoothing_type", sig.get("smoothing_type"), mapping=smoothing_map)
            # window size
            set_line("input_3D_window_size", sig.get("window_size"))
            # toggles
            set_check("chkbox_3D_normalize", sig.get("normalize"))
            set_check("chkbox_3D_average", sig.get("average"))
            set_check("chkbox_3D_cumulative", sig.get("cumulative"))

        # 3D Visualization -> Co-firing
        cof = (data or {}).get("3D Visualization", {}).get("Co-firing", {})
        if cof:
            set_check("cofiring_chkbox", cof.get("enabled"))
            set_line("cofiring_window_size", cof.get("window_size"))
            set_check("cofiring_shareA_chkbox", cof.get("shareA"))
            set_check("cofiring_shareB_chkbox", cof.get("shareB"))
            dir_map = {"bidirectional": "Bidirectional", "forward": "Forward", "backward": "Backward"}
            set_combo("cofiring_direction_dropdown", cof.get("direction"), mapping=dir_map)

        # 3D Visualization -> Shuffling
        shuf = (data or {}).get("3D Visualization", {}).get("Shuffling", {})
        if shuf:
            set_check("chkbox_shuffle_verified_only", shuf.get("verified_only"))
            set_check("chkbox_shuffle_spatial", shuf.get("spatial"))
            set_check("chkbox_shuffle_temporal", shuf.get("temporal"))
            set_line("shuffling_cofiring_window_size", shuf.get("cofiring_window_size"))
            set_check("shuffling_temporal_shareA_chkbox", shuf.get("shareA"))
            set_check("shuffling_temporal_shareB_chkbox", shuf.get("shareB"))
            dir_map2 = {"bidirectional": "Bidirectional", "forward": "Forward", "backward": "Backward"}
            set_combo("shuffling_cofiring_direction_dropdown", shuf.get("direction"), mapping=dir_map2)
            set_line("shuffle_num_shuffles", shuf.get("num_shuffles"))

        # Advanced Visualization -> FPR visualization
        fprv = (data or {}).get("Advanced Visualization", {}).get("FPR visualization", {})
        if fprv:
            set_line("input_3D_advanced_window_size", fprv.get("window_size"))
            set_combo("dropdown_3D_advanced_readout", fprv.get("readout"))
            set_combo("dropdown_3D_advanced_fpr", fprv.get("fpr"))
            set_slider("slider_3D_advanced_scaling", fprv.get("scaling"), 1, 1000)

        # Advanced Visualization -> FPR Shuffling
        fprs = (data or {}).get("Advanced Visualization", {}).get("FPR Shuffling", {})
        if fprs:
            set_check("visualization_3D_advanced_shuffle_spatial", fprs.get("spatial"))
            set_check("visualization_3D_advanced_shuffle_temporal", fprs.get("temporal"))
            set_line("input_advanced_shuffling_num", fprs.get("num_shuffles"))

        # Event based Shuffling
        ebs = (data or {}).get("Event based Shuffling", {})
        if ebs:
            set_combo("event_based_event_dropdown", ebs.get("event_type"))
            set_line("event_based_window_size_input", ebs.get("window_size"))
            set_line("event_based_lag_input", ebs.get("lag"))
            # slider
            set_slider("event_based_subwindows_slider", ebs.get("num_subwindows"), 1, 100)
            set_line("event_based_shuffles_input", ebs.get("num_shuffles"))
            set_check("event_based_amplitude_anchored", ebs.get("amplitude_anchored"))
            # Shuffle type mutual exclusion
            stype = self._scalar(ebs.get("shuffle_type"))
            if stype == "spatial":
                set_check("event_based_shuffle_spatial", True)
                set_check("event_based_shuffle_temporal", False)
            elif stype == "temporal":
                set_check("event_based_shuffle_spatial", False)
                set_check("event_based_shuffle_temporal", True)
            # which cells group
            set_combo("cmb_event_based_which_cells", ebs.get("which_cells"))
            # Ensure mutual exclusion logic runs (enables/disables amplitude anchored)
            try:
                owner.on_event_shuffle_type_changed()
            except Exception:
                pass

    def _on_browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Session Settings",
            "",
            "JSON Files (*.json)"
        )
        if not path:
            return
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self._apply_settings(data)
            self.accept()
        except Exception as e:
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Critical)
            msg.setWindowTitle("Load Failed")
            msg.setText("Failed to load session settings.")
            msg.setInformativeText(str(e))
            msg.exec_()