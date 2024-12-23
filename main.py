from caltrig.start_gui import start_gui

# Simple file to demonstrate how to start the application
# start_gui() needs to be contained within a __name__ == "__main__" block
# due to implementation of threading in the GUI.

if __name__ == "__main__":
    start_gui()