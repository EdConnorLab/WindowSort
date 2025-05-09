import os
import pickle
import re
from typing import Dict, List

import numpy as np
from PyQt5.QtWidgets import QFileDialog, QWidget, QInputDialog, QLineEdit
from scipy.signal import butter, filtfilt

from clat.intan.amplifiers import read_amplifier_data_with_mmap
from clat.intan.channels import Channel
from clat.intan.rhd import load_intan_rhd_format
from clat.intan.rhs import load_intan_rhs_format
from windowsort.drift import DriftingTimeAmplitudeWindow
from windowsort.units import Unit
from scipy.signal import sosfilt, butter



class InputDataManager:
    def __init__(self, intan_file_directory):
        self.intan_file_directory = intan_file_directory
        self.sample_rate = None
        self.amplifier_channels = None
        self.preprocessed_dir = os.path.join(intan_file_directory, "preprocessed_data")
        self.channel_cache = {}  # Limited memory cache for recently accessed channels
        self.max_cached_channels = 5  # Maximum number of channels to keep in memory
        self.init_data()

    def init_data(self):
        """Initialize by reading metadata but not loading all channel data"""
        # Load RHD/RHS info file
        if os.path.exists(os.path.join(self.intan_file_directory, "info.rhd")):
            info_path = os.path.join(self.intan_file_directory, "info.rhd")
            data = load_intan_rhd_format.read_data(info_path)
        elif os.path.exists(os.path.join(self.intan_file_directory, "info.rhs")):
            info_path = os.path.join(self.intan_file_directory, "info.rhs")
            data = load_intan_rhs_format.read_data(info_path)
        else:
            raise FileNotFoundError("No info.rhd or info.rhs file found in directory")

        # Store metadata
        self.amplifier_channels = data['amplifier_channels']
        self.sample_rate = data['frequency_parameters']['amplifier_sample_rate']

        # Ensure preprocessed directory exists
        if not os.path.exists(self.preprocessed_dir):
            os.makedirs(self.preprocessed_dir)

    def get_channel_data(self, channel: Channel) -> np.ndarray:
        """Get data for a specific channel, preprocessing if needed"""
        # Check if channel is in cache
        if channel in self.channel_cache:
            return self.channel_cache[channel]

        # Check if preprocessed data exists for this channel
        channel_file = os.path.join(self.preprocessed_dir, f"{channel.value}.npy")

        if os.path.exists(channel_file):
            # Load preprocessed data for this channel
            voltages = np.load(channel_file)
        else:
            # Preprocess this channel and save it
            voltages = self._preprocess_channel(channel)
            np.save(channel_file, voltages)

        # Add to cache, managing cache size
        self._update_cache(channel, voltages)

        return voltages

    def get_channels(self) -> List[Channel]:
        """Get list of available channels"""
        channels = []
        for ch_info in self.amplifier_channels:
            channel_name = ch_info.get("native_channel_name")
            if channel_name:
                channels.append(Channel(channel_name))
        return channels

    def _preprocess_channel(self, channel: Channel) -> np.ndarray:
        """Load and preprocess a single channel from the original data"""
        # Find channel index in amplifier_channels
        channel_idx = None
        for i, ch_info in enumerate(self.amplifier_channels):
            if ch_info.get("native_channel_name") == channel.value:
                channel_idx = i
                break

        if channel_idx is None:
            raise ValueError(f"Channel {channel.value} not found in amplifier channels")

        # Read only this channel's data from amplifier.dat
        amplifier_dat_path = os.path.join(self.intan_file_directory, "amplifier.dat")

        # Get file size to determine number of samples
        file_size = os.path.getsize(amplifier_dat_path)
        num_channels = len(self.amplifier_channels)
        num_samples = file_size // (num_channels * 2)  # int16 = 2 bytes

        # Memory-mapped read of just this channel
        with open(amplifier_dat_path, 'rb') as f:
            # Create memory mapping for the whole file
            data = np.memmap(amplifier_dat_path, dtype='int16', mode='r',
                             shape=(num_samples, num_channels))

            # Extract only the column for this channel
            channel_data = data[:, channel_idx].copy()  # Make a copy to avoid memmap issues

        # Convert to microvolts
        voltages = channel_data.astype(np.float32) * 0.195

        # Apply highpass filter
        filtered_voltages = self._highpass_filter_sos(voltages)

        return filtered_voltages

    def _highpass_filter(self, data, cutoff=300, order=5):
        """Apply highpass filter to data"""
        b, a = self._butter_highpass(cutoff, self.sample_rate, order=order)
        y = filtfilt(b, a, data)
        return y

    def _highpass_filter_sos(self, data, cutoff=300, order=5):
        """Apply highpass filter using second-order sections for better numerical stability"""
        sos = butter(order, cutoff / (0.5 * self.sample_rate), btype='high', output='sos')
        y = sosfilt(sos, data)
        return y

    def _butter_highpass(self, cutoff, fs, order=5):
        """Design highpass filter"""
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='high', analog=False)
        return b, a

    def _update_cache(self, channel: Channel, data: np.ndarray):
        """Add channel data to cache, removing least recently used if necessary"""
        # Add to cache
        self.channel_cache[channel] = data

        # If cache is too large, remove oldest entry
        if len(self.channel_cache) > self.max_cached_channels:
            # Remove first key (oldest) - this assumes Python 3.7+ where dict maintains insertion order
            oldest_channel = next(iter(self.channel_cache))
            del self.channel_cache[oldest_channel]

    def get_available_channels(self) -> List[Channel]:
        """Return list of all available channels"""
        return [Channel(ch.get("native_channel_name")) for ch in self.amplifier_channels]

    def clear_cache(self):
        """Clear the channel cache to free memory"""
        self.channel_cache.clear()


class SortedSpikeExporter:
    def __init__(self, *, save_directory):
        self.thresholded_spike_indices_by_channel = {}  # Keyed by channel, each value is a list of spike times
        self.sorted_spikes_by_unit_by_channel = {}  # Keyed by channel, each value is a dict of unit name to spike times
        self.save_directory = save_directory

    def update_thresholded_spikes(self, channel, thresholded_spike_indices):
        self.thresholded_spike_indices_by_channel[channel] = thresholded_spike_indices

    def save_thresholded_spikes(self):
        filename = os.path.join(self.save_directory, "thresholded_spikes.pkl")
        with open(filename, 'wb') as f:
            pickle.dump(self.thresholded_spike_indices_by_channel, f)
        # print(f"Saved {len(self.thresholded_spikes_by_channel.items())} thresholded spikes to {self.filename}")
        print(self.thresholded_spike_indices_by_channel)

    def save_sorted_spikes(self, spikes_by_unit: Dict[str, np.ndarray], channel, label=None):
        base_filename = "sorted_spikes"
        if label is not None:
            filename = base_filename + "_" + label + ".pkl"
        else:
            filename = base_filename + ".pkl"

        filename = os.path.join(self.save_directory, filename)

        # First, check if the file already exists.
        if os.path.exists(filename):
            # Load the existing data.
            with open(filename, 'rb') as f:
                existing_data = pickle.load(f)
        else:
            existing_data = {}

        # Update the specific channel's data in-memory.
        existing_data[channel] = spikes_by_unit

        # Now, save the updated data back to the file.
        with open(filename, 'wb') as f:
            pickle.dump(existing_data, f)

        for unit_name, spikes in spikes_by_unit.items():
            print(f"Saved {len(spikes)} spikes for unit {unit_name} to {filename}")


class SortingConfigManager:
    current_sorting_config_path: str = None

    def __init__(self, *, save_directory, voltage_time_plot, spike_plot, sort_panel, data_exporter):
        self.voltage_time_plot = voltage_time_plot
        self.spike_plot = spike_plot
        self.sort_panel = sort_panel
        self.data_exporter = data_exporter
        self.save_directory = save_directory
        self._set_current_sorting_config_path(os.path.join(self.save_directory, "sorting_config.pkl"))

    def open_current_sorting_config(self):
        channel = self.spike_plot.current_channel
        config = self._open_sorting_config(self.current_sorting_config_path, channel)
        self._apply_config(config)

    def open_selected_sorting_config(self):
        channel = self.spike_plot.current_channel
        print(f"Loading sorting config for channel {channel}")
        config = self._select_sorting_config(channel, self.sort_panel)
        self._apply_config(config)

    def save(self):
        channel = self.spike_plot.current_channel
        sorted_spikes_by_unit = self.sort_panel.sort_all_spikes(channel)

        file_label = self._get_current_file_label()
        # Use the DataExporter to save the sorted spikes
        self.data_exporter.save_sorted_spikes(sorted_spikes_by_unit, channel, label=file_label)
        self._save_sorting_config(channel, self.spike_plot.amp_time_windows,
                                  self.spike_plot.units,
                                  self.spike_plot.current_threshold_value,
                                  label=file_label)

    def save_as(self):
        channel = self.spike_plot.current_channel
        sorted_spikes_by_unit = self.sort_panel.sort_all_spikes(channel)

        file_label = self._query_file_label()
        print(file_label)

        # Use the DataExporter to save the sorted spikes
        self.data_exporter.save_sorted_spikes(sorted_spikes_by_unit, channel, label=file_label)
        self._save_sorting_config(channel, self.spike_plot.amp_time_windows,
                                  self.spike_plot.units,
                                  self.spike_plot.current_threshold_value,
                                  label=file_label)

    def _apply_config(self, config):
        if config:
            # Add threshold
            threshold = config['threshold']
            self.voltage_time_plot.update_threshold(threshold)
            self.voltage_time_plot.threshold_line.setValue(threshold)

            self.sort_panel.clear_all_unitpanels()
            self.spike_plot.clear_amp_time_windows()
            self.spike_plot.clear_units()

            # Add the amp time windows
            for window in config['amp_time_windows']:
                self.spike_plot.load_amp_time_window(window)

            self.unit_counter = 0
            for logical_expression, unit_name, color in config['units']:
                unit = Unit(logical_expression, unit_name, color)
                self.sort_panel.load_unit(unit)

            self.spike_plot.updatePlot()
            self.spike_plot.sortSpikes()

    def _query_file_label(self):
        # Open Input Dialog to get the filename extension
        text, ok = QInputDialog.getText(self.sort_panel, 'Input Dialog', 'Enter filename label:', QLineEdit.Normal, "")

        if ok and text:
            return text

    def _get_current_file_label(self):
        pattern = r"sorting_config_(.*?).pkl"
        match = re.search(pattern, self.current_sorting_config_path)
        return match.group(1) if match else None

    def _save_sorting_config(self, channel, amp_time_windows: List[DriftingTimeAmplitudeWindow], units, threshold,
                             label=None):
        base_filename = "sorting_config"
        if label is not None:
            filename = base_filename + "_" + label + ".pkl"
        else:
            filename = base_filename + ".pkl"
        filename = os.path.join(self.save_directory, filename)
        self._set_current_sorting_config_path(filename)

        try:
            with open(filename, 'rb') as f:
                all_configs = pickle.load(f)
        except FileNotFoundError:
            all_configs = {}

        all_configs[channel] = {
            'amp_time_windows': [window.time_control_points for window in amp_time_windows],
            'units': [(unit.logical_expression, unit.unit_name, unit.color) for unit in units],
            'threshold': threshold
        }

        with open(filename, 'wb') as f:
            pickle.dump(all_configs, f)

        print("Saved sorting configs to: ", filename)

    def _select_sorting_config(self, channel: Channel, parent_widget: QWidget):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        filename, _ = QFileDialog.getOpenFileName(parent_widget, "Open File", self.save_directory,
                                                  "Sorting Config Files (sorting_config*.pkl);;All Files (*)",
                                                  options=options)

        return self._open_sorting_config(filename, channel)

    def _set_current_sorting_config_path(self, filename):
        self.current_sorting_config_path = filename

    def _open_sorting_config(self, filename, channel):
        if filename:
            try:
                with open(filename, 'rb') as f:
                    all_configs = pickle.load(f)
                self._set_current_sorting_config_path(filename)
                return all_configs.get(channel, None)
            except FileNotFoundError:
                print(f"Configuration file {filename} not found.")
                return None
            except Exception as e:
                print(f"An error occurred while loading the configuration file: {e}")
                return None
