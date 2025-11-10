from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
                             QListWidget, QFileDialog, QLineEdit, QGroupBox, QMessageBox,
                             QListWidgetItem, QAbstractItemView)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
import json
import os


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
        group = QGroupBox("Analysis Parameter Files (JSON)")
        layout = QVBoxLayout()
        
        # Instructions
        info_label = QLabel(
            "Select a JSON file containing all analysis parameters.\n"
            "The file should include parameters for Co-Firing, Advanced Visualization, and Event-Based Shuffling."
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
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
        
        layout.addStretch()
        
        # Parameter status
        self.param_status_label = QLabel("Parameter file: Not selected")
        self.param_status_label.setStyleSheet("font-style: italic; margin-top: 5px;")
        layout.addWidget(self.param_status_label)
        
        group.setLayout(layout)
        return group
    
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
                
                # Update status label
                self.param_status_label.setText("Parameter file: Selected ✓")
                self.param_status_label.setStyleSheet("font-style: italic; margin-top: 5px; color: green;")
                
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
        """Run the automation (to be implemented)"""
        # Validate selections
        if not self.selected_sessions:
            QMessageBox.warning(self, "No Sessions", "Please select at least one session file.")
            return
        
        if not self.parameter_file:
            QMessageBox.warning(self, "No Parameters", "Please select a parameter file.")
            return
        
        # Show confirmation
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle("Automation Ready")
        msg.setText("Automation configuration complete!")
        msg.setInformativeText(
            f"Sessions: {len(self.selected_sessions)}\n"
            f"Parameter file: {os.path.basename(self.parameter_file)}\n\n"
            "Ready to implement automation logic."
        )
        msg.exec_()
        
        # TODO: Implement actual automation logic
        # This will be implemented later
        pass
    
    def get_automation_config(self):
        """Get the current automation configuration"""
        return {
            'sessions': self.selected_sessions,
            'parameter_file': self.parameter_file
        }
