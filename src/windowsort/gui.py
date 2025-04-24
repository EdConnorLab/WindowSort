import configparser
import os
import sys

from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QApplication, QHBoxLayout, QFileDialog, QMenuBar, QMenu, \
    QAction, QDialog, QLabel, QTextBrowser, QPushButton

from windowsort.datahandler import InputDataManager, SortedSpikeExporter, SortingConfigManager
from windowsort.drift import DriftSpikePlot, WindowMarkedSlider
from windowsort.snapshot import SnapshotPlot
from windowsort.spikes import SpikeScrubber
from windowsort.units import SortPanel
from windowsort.voltage import VoltageTimePlot, TimeScrubber, ChannelSelectionPanel, ThresholdControlPanel

import appdirs

# Use appdirs to get the appropriate user config directory
app_name = "WindowSort"
app_author = "EdConnorLab"
config_dir = appdirs.user_config_dir(app_name, app_author)
config_file = "app_config.ini"
CONFIG_PATH = os.path.join(config_dir, config_file)

# Ensure the config directory exists
os.makedirs(config_dir, exist_ok=True)


def main():
    app = QApplication(sys.argv)

    # Load the default directory
    # default_directory = get_default_directory()
    default_directory = "/home/r2_allen/Documents/EStimShape/allen_sort_250421_0"

    # Create a directory selection dialog
    options = QFileDialog.Options()
    options |= QFileDialog.ShowDirsOnly
    data_directory = QFileDialog.getExistingDirectory(None, "Select Data Directory", default_directory, options=options)

    # Check if the user pressed cancel (i.e., data_directory is empty)
    if not data_directory:
        print("No directory selected. Exiting.")
        sys.exit()

    # Save the selected directory as the new default
    save_default_directory(data_directory)

    print("Loading App")
    mainWin = MainWindow(data_directory)
    mainWin.showMaximized()
    sys.exit(app.exec_())


def save_default_directory(directory):
    """Save the parent directory of the given directory to the configuration file."""
    parent_directory = os.path.dirname(directory)

    config = configparser.ConfigParser()
    config.read(CONFIG_PATH)
    if 'DEFAULT' not in config:
        config['DEFAULT'] = {}
    config['DEFAULT']['data_directory'] = parent_directory
    with open(CONFIG_PATH, 'w') as configfile:
        config.write(configfile)


def get_default_directory():
    """Retrieve the default directory from the configuration file."""
    config = configparser.ConfigParser()
    config.read(CONFIG_PATH)
    return config.get('DEFAULT', 'data_directory', fallback='sftp://172.30.9.78/home/i2_allen/Documents/EStimShape/')


class MainWindow(QMainWindow):
    def __init__(self, data_directory):
        super(MainWindow, self).__init__()

        self.data_directory = data_directory
        # Initialize Dependencies
        self.data_handler = InputDataManager(data_directory)
        self.data_exporter = SortedSpikeExporter(save_directory=data_directory)
        self.sorting_config_manager = None

        # Initialize UI
        self.init_ui()
        self.init_menu_bar()

    def init_ui(self):
        self.setWindowTitle("Time Amp Window Sort GUI")

        central_widget = QWidget()
        main_layout = QHBoxLayout()

        # FIRST COLUMN
        self.setCentralWidget(central_widget)

        # Two Columns
        threshold_column = QVBoxLayout()
        spike_plot_column = QVBoxLayout()
        spike_sort_column = QVBoxLayout()

        # Voltage Time Plot
        self.voltage_time_plot = VoltageTimePlot(self.data_handler)
        threshold_column.addWidget(self.voltage_time_plot)
        self.time_scrubber = TimeScrubber(self.voltage_time_plot)
        threshold_column.addWidget(self.time_scrubber)
        self.threshold_control_panel = ThresholdControlPanel(self.voltage_time_plot)
        threshold_column.addWidget(self.threshold_control_panel)

        # Spike Plot
        default_max_spikes = 50
        self.spike_plot = DriftSpikePlot(self.data_handler,
                                         self.data_exporter,
                                         default_max_spikes=default_max_spikes)
        spike_plot_column.addWidget(self.spike_plot)
        self.voltage_time_plot.spike_plot = self.spike_plot
        self.spike_slider = WindowMarkedSlider(self.spike_plot)
        self.spike_scrubber = SpikeScrubber(self.spike_plot,
                                            default_max_spikes=default_max_spikes,
                                            slider=self.spike_slider)
        spike_plot_column.addWidget(self.spike_scrubber)
        self.spike_plot.spike_scrubber = self.spike_scrubber

        # Channel Selection
        self.channel_selection_pannel = ChannelSelectionPanel(self.voltage_time_plot, self.spike_plot)
        threshold_column.insertWidget(0, self.channel_selection_pannel)  # Inserts at the top of the layout

        # Sort Panel
        self.sort_panel = SortPanel(self.spike_plot,
                                    self.data_exporter,
                                    self.voltage_time_plot)
        spike_sort_column.insertWidget(0, self.sort_panel)
        self.spike_plot.set_sort_panel(self.sort_panel)
        self.channel_selection_pannel.sort_panel = self.sort_panel
        # Add more Time-Amp related widgets to spike_sort_layout if needed

        # Snapshot Plot
        self.snapshot_plot = SnapshotPlot(self.data_handler,
                                          self.data_exporter,
                                          self.sort_panel)
        self.snapshot_plot.setMaximumHeight(500)
        spike_plot_column.addWidget(self.snapshot_plot)

        # Sorting Manager
        self.sorting_config_manager = SortingConfigManager(save_directory=self.data_directory,
                                                           voltage_time_plot=self.voltage_time_plot,
                                                           spike_plot=self.spike_plot,
                                                           sort_panel=self.sort_panel,
                                                           data_exporter=self.data_exporter)
        self.channel_selection_pannel.sorting_config_manager = self.sorting_config_manager

        # Add the second column layout to the main layout
        main_layout.addLayout(threshold_column)
        main_layout.addLayout(spike_plot_column)
        main_layout.addLayout(spike_sort_column)

        central_widget.setLayout(main_layout)

    def init_menu_bar(self):
        # Menu Bar
        # Create a menu bar and add it to the main window
        menu_bar = QMenuBar(self)
        self.setMenuBar(menu_bar)
        self._file_menu(menu_bar)
        self._help_menu(menu_bar)

    def _file_menu(self, menu_bar):
        # Create a "File" menu and add it to the menu bar
        file_menu = QMenu("File", self)
        menu_bar.addMenu(file_menu)
        # Create actions
        open_action = QAction("Open Config (Ctrl+O)", self)
        open_action.setShortcut("Ctrl+O")
        save_action = QAction("Save (Ctrl+S)", self)
        save_action.setShortcut("Ctrl+S")
        save_as_action = QAction("Save As... (Ctrl+Shift+S)", self)
        save_as_action.setShortcut("Ctrl+Shift+S")
        # Add actions to the file menu
        file_menu.addAction(open_action)
        file_menu.addAction(save_action)
        file_menu.addAction(save_as_action)
        # Connect actions to functions
        open_action.triggered.connect(
            self.sorting_config_manager.open_selected_sorting_config)  # replace with the correct function
        save_action.triggered.connect(self.sorting_config_manager.save)  # replace with the correct function
        save_as_action.triggered.connect(self.sorting_config_manager.save_as)  # replace with the correct function

    def _help_menu(self, menu_bar):
        help_menu = self.menuBar().addMenu("Help")
        menu_bar.addMenu(help_menu)

        # Add Controls action to the Help menu
        controls_action = QAction("Controls", self)
        controls_action.triggered.connect(self.show_controls_dialog)
        help_menu.addAction(controls_action)

    def show_controls_dialog(self):
        dialog = ControlsDialog(self)
        dialog.exec_()


class ControlsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Controls")

        layout = QVBoxLayout(self)

        # Main Heading
        main_heading = QLabel("<h2>Application Controls</h2>")
        layout.addWidget(main_heading)

        # Section 1: General Controls
        section1_heading = QLabel("<h3>General Controls</h3>")
        layout.addWidget(section1_heading)

        general_controls_text = """
        <ul>
            <li><b>Ctrl + S:</b> Save</li>
            <li><b>Ctrl + Shift + S:</b> Save As...</li>
            <li><b>Ctrl + O:</b> Open Sorting Config</li>
        </ul>
        """
        general_controls_browser = QTextBrowser(self)
        general_controls_browser.setHtml(general_controls_text)
        layout.addWidget(general_controls_browser)

        # Section 2: Window Controls
        section2_heading = QLabel("<h3>Time-Amplitude Window Controls</h3>")
        layout.addWidget(section2_heading)

        window_controls_text = """
        <ul>
            <li><b>SHIFT + Left Click:</b> Create Window</li>
            <li><b>Left Click:</b> Select Window</li>
            <li><b>Delete:</b> Delete Selected Window</li>
            <li><b>WASD or Left Click Hold and Drag:</b> Move Selected Window</li>
            <li><b>Up or Down Arrow Keys:</b> Size Up or Down Selected Window</li>
            <li><b>Left or Right Arrow Keys:</b> Step to Previous or Next Spikes</li>
            <li><b>SHIFT + Left or Right Arrow Keys:</b> Large Step to Previous or Next Spikes</li>
            <li><b>SPACE:</b> Create Time Control Point for Selected Window</li>
            <li><b>Backspace:</b> Delete Time Control Point for Selected Window</li>
            <li><b>Double Click on Colored Ticks on Spike Scrubber:</b> Navigate to Selected Time Control Point</li>
        </ul>
        """
        window_controls_browser = QTextBrowser(self)
        window_controls_browser.setHtml(window_controls_text)
        layout.addWidget(window_controls_browser)

        # Section 3: Time Control Point Controls
        section3_heading = QLabel("<h3>Time Control Point Controls</h3>")
        layout.addWidget(section3_heading)
        time_control_points_text = """
        <ul>
            <li><b>SPACE:</b> Create Time Control Point for Selected Window</li>
            <li><b>Backspace:</b> Delete Time Control Point for Selected Window</li>
            <li><b>Double Click on Colored Ticks on Spike Scrubber:</b> Colored ticks correspond to starting times of time control points. Navigate to selected Time Control Point</li>
            <li><b>Info:</b> Time control points determine the location and height of a window for a specific time point in the experiment. 
            A spike will be sorted using the window corresponding to the closest proceeding time control point to the spike's time. 
            </li>
        </ul>
        """
        time_control_points_browser = QTextBrowser(self)
        time_control_points_browser.setHtml(time_control_points_text)
        layout.addWidget(time_control_points_browser)

        # Close button
        close_button = QPushButton("Close", self)
        close_button.clicked.connect(self.close)
        layout.addWidget(close_button)

        self.setLayout(layout)


if __name__ == '__main__':
    main()
