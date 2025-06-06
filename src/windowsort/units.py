from __future__ import annotations

import itertools
from typing import List
import re

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox, QFrame, QInputDialog

from windowsort.colors import unit_color_generator
from windowsort.threshold import threshold_spikes_absolute
from windowsort.voltage import VoltageTimePlot


class Unit:
    def __init__(self, logical_expression, unit_name, color):
        self.logical_expression = logical_expression  # Expression can be something like "W1 and not W2"
        self.unit_name = unit_name  # Unit identifier, e.g., "Unit 1"
        self.color = color  # Color for the unit, e.g., 'green'

    def sort_spike(self, *, voltage_index_of_spike, spike_number, voltages, amp_time_windows):
        """
        
        :param voltage_index_of_spike: the index in voltage array where the spike occured 
        :param spike_number: number spike it is in the experiment (e.g., 1st, 2nd, 3rd, etc.)
        :param voltages: 
        :param amp_time_windows: 
        :return: 
        """
        # Generate a dictionary of window results
        window_results = {}
        for idx, window in enumerate(amp_time_windows):
            window_results[f'w{idx + 1}'] = window.is_spike_in_window(voltage_index_of_spike, spike_number, voltages)

        # Try to evaluate the logical expression
        try:
            python_compatible_expression = self.logical_expression.lower()
            python_compatible_expression = self.upper_case_true_false(python_compatible_expression)
            python_compatible_expression = self.process_ignore_operators(python_compatible_expression)

            result = eval(python_compatible_expression, {}, window_results)
        except Exception as e:
            print(f"Invalid expression: {self.logical_expression}, Error: {e}")
            result = False

        return result

    @staticmethod
    def upper_case_true_false(expression: str) -> str:
        processed_expression = expression
        # Find all occurrences of "True" and replace with "True"
        if "true" in processed_expression:
            processed_expression = processed_expression.replace("true", "True")

        # Find all occurrences of "False" and replace with "False"
        if "false" in processed_expression:
            processed_expression = processed_expression.replace("false", "False")

        return processed_expression

    @staticmethod
    def process_ignore_operators(expression: str) -> str:
        processed_expression = expression
        # Find all occurrences of "IGNORE Wx" and replace with "True AND"
        for i in range(1, 10):  # Assuming up to W9 for this example
            ignore_str = f"ignore w{i}"
            if ignore_str in processed_expression:
                processed_expression = processed_expression.replace(ignore_str, "and True")

        return processed_expression


class SortPanel(QWidget):
    unit_panels: List[UnitPanel]

    def __init__(self, thresholded_spike_plot, data_exporter,
                 voltage_time_plot: VoltageTimePlot):
        super(SortPanel, self).__init__(thresholded_spike_plot)
        self.spike_plot = thresholded_spike_plot
        self.data_exporter = data_exporter
        self.voltage_time_plot = voltage_time_plot
        self.unit_panels = []
        self.unit_counter = 0  # to generate unique unit identifiers
        self.current_color = None
        self.unit_colors = unit_color_generator()  # to generate unique colors for units
        self.layout = QVBoxLayout()

        # Add a button for adding new units
        self.add_unit_button = QPushButton("Add New Unit")
        self.add_unit_button.clicked.connect(self.add_new_unit)
        self.layout.addWidget(self.add_unit_button)

        # Add a legend for window colors
        self.legend_layout = QVBoxLayout()
        self.legend_container = QWidget()
        self.legend_container.setLayout(self.legend_layout)
        self.legend_container.setMaximumHeight(200)  # Set maximum height
        self.layout.addWidget(self.legend_container)

        # Unit panels
        self.unit_panels_layout = QVBoxLayout()
        self.layout.addLayout(self.unit_panels_layout)

        self.previous_window_count = 0

        self.setLayout(self.layout)

    def _update_legend(self):
        """Update the legend to match the current window colors."""

        # Clear the existing legend
        for i in reversed(range(self.legend_layout.count())):
            layout_item = self.legend_layout.itemAt(i)
            if layout_item.layout() is not None:
                # Delete all widgets from the layout
                for j in reversed(range(layout_item.layout().count())):
                    widget = layout_item.layout().itemAt(j).widget()
                    if widget is not None:
                        widget.deleteLater()
                # Remove the layout itself
                layout_item.layout().deleteLater()

        # Add new color squares and text to the legend
        window_colors = self.get_window_colors()
        for idx, color in enumerate(window_colors):
            unit_row_layout = QHBoxLayout()
            unit_row_layout.setSpacing(10)  # Reduce the spacing between widgets within the row
            unit_row_layout.setContentsMargins(0, 0, 0, 0)  # Reduce the margins for this row

            square = QFrame()
            square.setFixedSize(10, 10)
            square.setStyleSheet(f"QWidget {{ background-color: {color}; }}")
            unit_row_layout.addWidget(square)

            label = QLabel(f"W{idx + 1}")
            unit_row_layout.addWidget(label)

            self.legend_layout.addLayout(unit_row_layout)

    def emit_recalculate_windows(self):
        self.spike_plot.windowUpdated.emit()

    def add_new_unit(self):
        self.unit_counter += 1
        self.current_color = next(self.unit_colors)
        new_unit_panel = UnitPanel(self.unit_counter, self.current_color, self.unit_panels_layout, self.spike_plot,
                                   self.delete_unit)

        new_unit_panel.populate_with_new_unit()
        self.unit_panels.append(new_unit_panel)

    def load_unit(self, unit: Unit):
        self.unit_counter += 1
        self.current_color = next(self.unit_colors)
        new_unit_panel = UnitPanel(self.unit_counter, self.current_color, self.unit_panels_layout, self.spike_plot,
                                   self.delete_unit)

        new_unit_panel.expression = unit.logical_expression
        new_unit_panel.populate_with_existing_unit(unit)
        new_unit_panel.expression_line_edit.setText(unit.logical_expression)

        self.unit_panels.append(new_unit_panel)

    def on_window_number_change(self, deleted_window_number=None):
        """Update expressions in each unit layout to match the current number of windows."""
        if deleted_window_number is not None:
            # Detect if the number of windows decreased:
            self._on_window_deleted(deleted_window_number)

        self._update_legend()

        for unit_panel in self.unit_panels:
            unit_panel.update_expression_from_text_box()

    def _on_window_deleted(self, deleted_window_number: int):
        """
        Handle updates after a window is deleted.

        :param deleted_window_number: The number of the deleted window.
        """
        for unit_panel in self.unit_panels:
            old_expression = unit_panel.expression
            new_expression = self._generate_new_expression_for_deletion(old_expression, deleted_window_number)

            # Update internal expression
            unit_panel.unit.logical_expression = new_expression

            # Update displayed expression
            if isinstance(unit_panel, UnitPanel):
                # If using UnitPanel, update the QLineEdit text
                unit_panel.expression_line_edit.setText(new_expression)

    def _generate_new_expression_for_deletion(self, old_expression: str, deleted_window: int) -> str:
        """
        Update a logical expression after a window is deleted.

        :param old_expression: The original expression to be updated.
        :param deleted_window: The number of the deleted window.
        :return: The updated expression.
        """
        # Pattern to find window references like "W1", "W2", etc.
        # can be lower case as well
        pattern = re.compile(r"W(\d+)", re.IGNORECASE)

        def replacer(match):
            # Extract the window number
            window_number = int(match.group(1))

            # If the window number is higher than the deleted window, decrement it
            if window_number > deleted_window:
                return f"W{window_number - 1}"
            # If the window number is equal to the deleted window, replace it with "False"
            elif window_number == deleted_window:
                return "True"
            # Otherwise, keep it as is
            else:
                return f"W{window_number}"

        # Replace all window references in the old expression
        new_expression = pattern.sub(replacer, old_expression)
        print(new_expression)
        return new_expression

    def _detect_deleted_window(self):
        """
        Detect which window number was deleted by finding a gap in the sequence.

        :return: The number of the deleted window.
        """
        # Create a list to store the window numbers
        window_numbers = []

        # Loop through all windows and extract their numbers
        for window_index, window in enumerate(self.spike_plot.amp_time_windows):
            # Assume window labels are in the form "W1", "W2", etc.
            window_number = window_index + 1
            window_numbers.append(window_number)

        # Find the gap in the sequence of window numbers
        window_numbers.sort()
        for idx, num in enumerate(window_numbers):
            if idx + 1 != num:
                # We found a gap! The deleted window number is idx + 1
                return idx + 1

        # If no gap was found, return None or handle this case as appropriate for your application
        return None

    def get_window_colors(self):
        window_colors = [window.color for window in self.spike_plot.amp_time_windows]
        return window_colors

    def delete_unit(self, unit_panel):
        """Delete a specific unit."""
        # Remove unit from the list
        self.unit_panels.remove(unit_panel)

        # Actually remove the unit from data
        self.spike_plot.removeUnit(unit_panel.unit.unit_name)

        # Clear and delete layout
        unit_panel.clear_unit_layout()
        del unit_panel

        # Repopulate remaining units
        self.on_window_number_change()

        self.unit_counter -= 1

        self.spike_plot.sortSpikes()

    def clear_all_unitpanels(self):
        """Delete all units."""
        to_delete = []
        for unit_panel in self.unit_panels:
            to_delete.append(unit_panel)
        for unit_panel in to_delete:
            self.delete_unit(unit_panel)
        self.unit_colors = unit_color_generator()
        self.unit_counter = 0


    def sort_all_spikes(self, channel):
        voltages = self.spike_plot.data_handler.get_channel_data(channel)
        self.spike_plot.updatePlot()  # Make sure the data is updated
        sorted_spikes_by_unit = {}
        # Pre-compute the crossing indices for the current channel
        threshold_value = self.spike_plot.current_threshold_value
        crossing_indices = threshold_spikes_absolute(threshold_value, voltages)
        for unit in self.spike_plot.units:
            sorted_spikes = []  # List to hold the sorted spikes for this unit

            for crossing_point_index, point in enumerate(crossing_indices):
                spike_voltage_index = round(point)  # index in the voltage data

                # Check if the spike belongs to the current unit
                if unit.sort_spike(voltage_index_of_spike=spike_voltage_index, spike_number=crossing_point_index,
                                   voltages=voltages,
                                   amp_time_windows=self.spike_plot.amp_time_windows):
                    sorted_spikes.append(spike_voltage_index)

            sorted_spikes_by_unit[unit.unit_name] = sorted_spikes
        return sorted_spikes_by_unit


def create_wrapped_label(unit_name_label):
    wrapper = QWidget()
    wrapper_layout = QVBoxLayout()
    wrapper_layout.addWidget(unit_name_label, alignment=Qt.AlignCenter)
    wrapper.setLayout(wrapper_layout)
    return wrapper


from PyQt5.QtWidgets import QLineEdit


class UnitPanel:
    """
    subpanel within SortPanel that controls a single unit
    """

    def __init__(self, unit_counter, unit_color, parent_layout, thresholded_spike_plot, delete_func):
        self.unit = None
        self.unit_counter = unit_counter
        self.unit_color = unit_color
        self.parent_layout = parent_layout
        self.delete_func = delete_func
        self.spike_plot = thresholded_spike_plot
        self.expression_line_edit = None  # The text box for manually entering expressions
        self.expression = None

    def populate_with_new_unit(self):
        self.unit_layout = QHBoxLayout()

        self.create_unit()
        self.add_delete_button()
        self.add_unit_label()

        self.populate_unit_panel()  # Add the text box for manually entering expressions

        self.parent_layout.addLayout(self.unit_layout)

    def populate_with_existing_unit(self, unit):
        self.unit_layout = QHBoxLayout()
        self.unit = unit
        self.spike_plot.addUnit(self.unit)

        self.add_delete_button()
        self.add_unit_label()

        self.populate_unit_panel()  # Add the text box for manually entering expressions

        self.parent_layout.addLayout(self.unit_layout)

    def add_unit_label(self):
        label = QLabel(self.unit.unit_name)
        label.setStyleSheet(f"background-color: {self.unit_color};")
        wrapper = create_wrapped_label(label)
        self.unit_layout.addWidget(wrapper)

    def add_delete_button(self):
        delete_button = QPushButton("Delete Unit")
        delete_button.clicked.connect(lambda: self.delete_func(self))
        self.unit_layout.addWidget(delete_button)

    def populate_unit_panel(self):
        self.expression_line_edit = QLineEdit()
        self.expression_line_edit.setPlaceholderText("Enter expression (e.g., W1 and not W2)")
        self.expression_line_edit.editingFinished.connect(self.update_expression_from_text_box)
        try:
            print("Expression: " + self.expression)
        except:
            print("Expression not set yet")
        self.unit_layout.addWidget(self.expression_line_edit)

    def update_expression_from_text_box(self):
        self.expression = self.expression_line_edit.text()
        updated_unit = Unit(self.expression, self.unit.unit_name, self.unit_color)
        print(f"Updating unit {self.unit.unit_name} with expression {self.expression}")
        self.spike_plot.updateUnit(updated_unit)
        self.spike_plot.sortSpikes()

    def create_unit(self):
        self.unit = Unit(self.expression, f"Unit {self.unit_counter}", self.unit_color)
        self.spike_plot.addUnit(self.unit)

    def clear_unit_layout(self):
        """Clear existing widgets from the unit layout."""
        for i in reversed(range(self.unit_layout.count())):
            widget = self.unit_layout.itemAt(i).widget()
            if widget is not None:
                # Remove widget from layout and delete
                self.unit_layout.removeWidget(widget)
                widget.deleteLater()
        self.parent_layout.removeItem(self.unit_layout)

