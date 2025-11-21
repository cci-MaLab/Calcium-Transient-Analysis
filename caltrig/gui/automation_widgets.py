from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
                             QListWidget, QFileDialog, QLineEdit, QGroupBox, QMessageBox,
                             QListWidgetItem, QAbstractItemView, QProgressDialog, QProgressBar, QCheckBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QIcon
import json
import os
from ..core.automation import run_batch_automation


class AutomationDialog(QDialog):
    """
    Dialog window for automating analysis output across multiple sessions with different parameters.
    
    Allows users to:
    1. Select multiple session configuration files
    2. Select parameter JSON files for different analyses
    3. Run automated analysis (to be implemented)
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Automate Analysis Output")
        self.setMinimumWidth(800)
        self.setMinimumHeight(600)
        
        # Store selected files
        self.selected_sessions = []
        self.parameter_file = None  # Single JSON file containing all parameters
        self.output_path = None  # Output directory (None = use local session directories)
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize the user interface"""
        main_layout = QVBoxLayout()
        
        # Title
        title_label = QLabel("Automated Analysis Output")
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; margin: 10px;")
        main_layout.addWidget(title_label)
        
        # Description
        desc_label = QLabel(
            "This tool allows you to run multiple analyses across different sessions with varying parameters.\n"
            "Select session files and parameter configurations, then click 'Run Automation' to generate outputs."
        )
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("margin: 5px; padding: 10px; background-color: #f0f0f0;")
        main_layout.addWidget(desc_label)
        
        # Main content area (horizontal split)
        content_layout = QHBoxLayout()
        
        # Left side: Session selection
        session_group = self.create_session_selection_group()
        content_layout.addWidget(session_group, stretch=1)
        
        # Right side: Parameter files
        parameter_group = self.create_parameter_selection_group()
        content_layout.addWidget(parameter_group, stretch=1)
        
        main_layout.addLayout(content_layout)
        
        # Output path selection
        output_group = self.create_output_path_group()
        main_layout.addWidget(output_group)
        
        # Bottom buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        self.btn_run = QPushButton("Run Automation")
        self.btn_run.clicked.connect(self.run_automation)
        self.btn_run.setEnabled(False)  # Disabled until both sessions and parameters are selected
        self.btn_run.setStyleSheet("font-weight: bold; padding: 8px 20px;")
        button_layout.addWidget(self.btn_run)
        
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.reject)
        button_layout.addWidget(btn_close)
        
        main_layout.addLayout(button_layout)
        
        self.setLayout(main_layout)
    
    def create_session_selection_group(self):
        """Create the session selection group"""
        group = QGroupBox("Session Configuration Files")
        layout = QVBoxLayout()
        
        # Instructions
        info_label = QLabel("Select session configuration (.ini) files to process:")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        # List widget for sessions
        self.session_list = QListWidget()
        self.session_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        layout.addWidget(self.session_list)
        
        # Buttons for session management
        session_btn_layout = QHBoxLayout()
        
        btn_add_session = QPushButton("Add Session(s)")
        btn_add_session.clicked.connect(self.add_sessions)
        session_btn_layout.addWidget(btn_add_session)
        
        btn_remove_session = QPushButton("Remove Selected")
        btn_remove_session.clicked.connect(self.remove_sessions)
        session_btn_layout.addWidget(btn_remove_session)
        
        btn_clear_sessions = QPushButton("Clear All")
        btn_clear_sessions.clicked.connect(self.clear_sessions)
        session_btn_layout.addWidget(btn_clear_sessions)
        
        layout.addLayout(session_btn_layout)
        
        # Session count label
        self.session_count_label = QLabel("Sessions selected: 0")
        self.session_count_label.setStyleSheet("font-style: italic; margin-top: 5px;")
        layout.addWidget(self.session_count_label)
        
        group.setLayout(layout)
        return group
    
    def create_parameter_selection_group(self):
        """Create the parameter file selection group"""
        group = QGroupBox("Analysis Parameters")
        layout = QVBoxLayout()
        
        # Single parameter file selection
        param_layout = QHBoxLayout()
        param_layout.addWidget(QLabel("Parameters File:"))
        self.param_input = QLineEdit()
        self.param_input.setReadOnly(True)
        self.param_input.setPlaceholderText("No file selected")
        param_layout.addWidget(self.param_input, stretch=1)
        btn_param = QPushButton("Browse...")
        btn_param.clicked.connect(self.select_parameter_file)
        param_layout.addWidget(btn_param)
        layout.addLayout(param_layout)
        
        # Output type selection checkboxes
        layout.addWidget(QLabel("Output Types:"))
        self.chk_cofiring = QCheckBox("Co-firing")
        self.chk_cofiring.setChecked(True)
        layout.addWidget(self.chk_cofiring)
        
        self.chk_advanced = QCheckBox("Advanced")
        self.chk_advanced.setChecked(True)
        layout.addWidget(self.chk_advanced)
        
        self.chk_event_based = QCheckBox("Event-based")
        self.chk_event_based.setChecked(True)
        layout.addWidget(self.chk_event_based)
        
        layout.addStretch()
        
        group.setLayout(layout)
        return group
    
    def create_output_path_group(self):
        """Create the output path selection group"""
        group = QGroupBox("Output Directory")
        layout = QVBoxLayout()
        
        # Path input and browse button
        path_layout = QHBoxLayout()
        
        self.output_path_input = QLineEdit()
        self.output_path_input.setPlaceholderText("Default: Current directory")
        self.output_path_input.setReadOnly(True)
        path_layout.addWidget(self.output_path_input, stretch=1)
        
        btn_browse_output = QPushButton("Browse...")
        btn_browse_output.clicked.connect(self.select_output_path)
        path_layout.addWidget(btn_browse_output)
        
        btn_clear_output = QPushButton("Clear")
        btn_clear_output.clicked.connect(self.clear_output_path)
        path_layout.addWidget(btn_clear_output)
        
        layout.addLayout(path_layout)
        
        group.setLayout(layout)
        return group
    
    def select_output_path(self):
        """Select output directory"""
        path = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory",
            ""
        )
        
        if path:
            self.output_path = path
            self.output_path_input.setText(path)
    
    def clear_output_path(self):
        """Clear output path (use default)"""
        self.output_path = None
        self.output_path_input.clear()
    
    def add_sessions(self):
        """Add session configuration files"""
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Session Configuration Files",
            "",
            "Configuration Files (*.ini);;All Files (*)"
        )
        
        if files:
            for file_path in files:
                # Check if already added
                if file_path not in self.selected_sessions:
                    self.selected_sessions.append(file_path)
                    # Add to list widget
                    item = QListWidgetItem(os.path.basename(file_path))
                    item.setData(Qt.UserRole, file_path)  # Store full path
                    item.setToolTip(file_path)  # Show full path on hover
                    self.session_list.addItem(item)
            
            self.update_session_count()
            self.check_ready_to_run()
    
    def remove_sessions(self):
        """Remove selected sessions from the list"""
        for item in self.session_list.selectedItems():
            file_path = item.data(Qt.UserRole)
            self.selected_sessions.remove(file_path)
            self.session_list.takeItem(self.session_list.row(item))
        
        self.update_session_count()
        self.check_ready_to_run()
    
    def clear_sessions(self):
        """Clear all sessions"""
        self.selected_sessions.clear()
        self.session_list.clear()
        self.update_session_count()
        self.check_ready_to_run()
    
    def update_session_count(self):
        """Update the session count label"""
        count = len(self.selected_sessions)
        self.session_count_label.setText(f"Sessions selected: {count}")
    
    def select_parameter_file(self):
        """Select the parameter JSON file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Analysis Parameters File",
            "",
            "JSON Files (*.json);;All Files (*)"
        )
        
        if file_path:
            # Validate JSON file
            try:
                with open(file_path, 'r') as f:
                    json.load(f)
                
                # Store the file path
                self.parameter_file = file_path
                
                # Update the input field
                self.param_input.setText(os.path.basename(file_path))
                self.param_input.setToolTip(file_path)
                
                self.check_ready_to_run()
                
            except json.JSONDecodeError:
                QMessageBox.warning(
                    self,
                    "Invalid JSON",
                    f"The selected file is not a valid JSON file:\n{file_path}"
                )
            except Exception as e:
                QMessageBox.warning(
                    self,
                    "Error",
                    f"Error reading file:\n{str(e)}"
                )
    
    def check_ready_to_run(self):
        """Check if ready to run automation and enable/disable run button"""
        has_sessions = len(self.selected_sessions) > 0
        has_params = self.parameter_file is not None
        
        self.btn_run.setEnabled(has_sessions and has_params)
    
    def run_automation(self):
        """Run the automation"""
        # Validate selections
        if not self.selected_sessions:
            QMessageBox.warning(self, "No Sessions", "Please select at least one session file.")
            return
        
        if not self.parameter_file:
            QMessageBox.warning(self, "No Parameters", "Please select a parameter file.")
            return
        
        # Create progress dialog
        progress = QProgressDialog("Starting automation...", "Cancel", 0, 100, self)
        progress.setWindowTitle("Automation Progress")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setMinimumWidth(400)
        
        # Track progress with labels
        session_label = QLabel("Session: Initializing...")
        analysis_label = QLabel("Analysis: Waiting...")
        
        def update_session(current, total, session_name):
            session_label.setText(f"Session: {session_name} ({current}/{total})")
            progress.setLabelText(f"Session: {session_name} ({current}/{total})\n{analysis_label.text()}")
            progress.setValue(int((current / total) * 100))
        
        def update_analysis(analysis_type):
            analysis_label.setText(f"Analysis: {analysis_type}")
            progress.setLabelText(f"{session_label.text()}\n{analysis_label.text()}")
        
        def analysis_complete():
            pass  # Could update if needed
        
        # Get which outputs are enabled
        enabled_outputs = {
            'cofiring': self.chk_cofiring.isChecked(),
            'advanced': self.chk_advanced.isChecked(),
            'event_based': self.chk_event_based.isChecked()
        }
        
        try:
            results = run_batch_automation(
                self.selected_sessions, 
                self.parameter_file, 
                self.output_path,
                enabled_outputs=enabled_outputs,
                progress_callback_session=update_session,
                progress_callback_analysis=update_analysis,
                progress_callback_analysis_done=analysis_complete
            )
            
            progress.close()
            
            # Show results
            success_count = len(results['successful'])
            failed_count = len(results['failed'])
            
            if failed_count > 0:
                error_details = '\n'.join([f"{r['path']}: {r['error']}" for r in results['failed']])
                QMessageBox.warning(
                    self,
                    "Automation Complete with Errors",
                    f"Processed: {success_count} successful, {failed_count} failed\n\n{error_details}"
                )
            else:
                QMessageBox.information(
                    self,
                    "Automation Complete",
                    f"Successfully processed {success_count} session(s)"
                )
                
        except Exception as e:
            QMessageBox.critical(
                self,
                "Automation Error",
                f"Failed to run automation:\n{str(e)}"
            )
            import traceback
            traceback.print_exc()
        pass
    
    def get_automation_config(self):
        """Get the current automation configuration"""
        return {
            'sessions': self.selected_sessions,
            'parameter_file': self.parameter_file,
            'output_path': self.output_path
        }
