from caltrig.start_gui import start_gui
import argparse

# Simple file to demonstrate how to start the application
# start_gui() needs to be contained within a __name__ == "__main__" block
# due to implementation of threading in the GUI.

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start the GUI application.")
    parser.add_argument(
        "--no-processes",
        action="store_true",  # Makes it a flag (True if specified)
        help="Disable processes in the GUI. Helps with some very slow systems. That struggle with multiprocessing.",
    )
    args = parser.parse_args()
    
    # Set processes to False if --no-processes is specified
    start_gui(processes=not args.no_processes)