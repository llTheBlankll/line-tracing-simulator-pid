"""
Visualization module for the line following robot simulation.
Contains the MatplotlibWindow class for displaying robot data in real-time plots.
"""
import tkinter as tk
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import threading
import time

class MatplotlibWindow:
    """
    A class that creates a separate window with Matplotlib plots
    to display robot performance data in real-time.
    """
    def __init__(self, data_queue):
        """
        Initialize the Matplotlib window.

        Args:
            data_queue: A queue for receiving data from the main simulation
        """
        self.data_queue = data_queue
        self.is_closed = False
        self.check_queue_id = None
        
        # Create Tkinter window
        self.root = tk.Tk()
        self.root.title("Robot Performance")
        self.root.geometry("800x600")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Create a frame to hold the plots
        frame = ttk.Frame(self.root)
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Create Figure and subplots
        self.fig = Figure(figsize=(8, 6), dpi=100)
        
        # Create subplots
        self.error_ax = self.fig.add_subplot(2, 2, 1)
        self.error_ax.set_title('Error')
        self.error_ax.set_ylim(-1.5, 1.5)
        self.error_line, = self.error_ax.plot([], [], 'r-')
        
        self.motor_ax = self.fig.add_subplot(2, 2, 2)
        self.motor_ax.set_title('Motor Speeds')
        self.motor_ax.set_ylim(-10, 100)
        self.left_motor_line, = self.motor_ax.plot([], [], 'b-', label='Left')
        self.right_motor_line, = self.motor_ax.plot([], [], 'g-', label='Right')
        self.motor_ax.legend()
        
        self.pid_ax = self.fig.add_subplot(2, 2, 3)
        self.pid_ax.set_title('PID Components')
        self.pid_ax.set_ylim(-20, 20)
        self.p_line, = self.pid_ax.plot([], [], 'r-', label='P')
        self.i_line, = self.pid_ax.plot([], [], 'g-', label='I')
        self.d_line, = self.pid_ax.plot([], [], 'b-', label='D')
        self.pid_ax.legend()
        
        self.diff_ax = self.fig.add_subplot(2, 2, 4)
        self.diff_ax.set_title('Motor Speed Difference')
        self.diff_ax.set_ylim(-40, 40)
        self.diff_line, = self.diff_ax.plot([], [], 'k-')
        
        # Add tight layout
        self.fig.tight_layout()
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initialize data
        self.x_data = np.arange(100)
        self.error_data = np.zeros(100)
        self.left_motor_data = np.zeros(100)
        self.right_motor_data = np.zeros(100)
        self.p_data = np.zeros(100)
        self.i_data = np.zeros(100)
        self.d_data = np.zeros(100)
        self.diff_data = np.zeros(100)
        
        # Start periodic data check - using named callback function
        self.check_queue()
        
    def check_queue(self):
        """
        Check the data queue for new data and update plots.
        This method is called periodically using Tkinter's after method.
        """
        if self.is_closed:
            return
            
        try:
            # Check if there's new data in the queue
            if not self.data_queue.empty():
                data = self.data_queue.get(block=False)
                self.update_plots(data)
                
            # Schedule this method to run again after 100ms
            if not self.is_closed:
                # Cancel any existing scheduled call
                if self.check_queue_id is not None:
                    self.root.after_cancel(self.check_queue_id)
                # Schedule a new call
                self.check_queue_id = self.root.after(100, self.check_queue)
        except Exception as e:
            print(f"Error in MatplotlibWindow.check_queue: {e}")
            if not self.is_closed:
                # Cancel any existing scheduled call
                if self.check_queue_id is not None:
                    self.root.after_cancel(self.check_queue_id)
                # Schedule a new call
                self.check_queue_id = self.root.after(100, self.check_queue)
    
    def update_plots(self, data):
        """
        Update all plots with new data.
        
        Args:
            data: Dictionary containing the data to plot
        """
        try:
            # Extract data from the dictionary
            error_history = data.get('error_history', [])
            left_speed_history = data.get('left_speed_history', [])
            right_speed_history = data.get('right_speed_history', [])
            proportional_history = data.get('proportional_history', [])
            integral_history = data.get('integral_history', [])
            derivative_history = data.get('derivative_history', [])
            motor_diff_history = data.get('motor_diff_history', [])
            
            # Ensure we have data to plot
            if not error_history:
                return
                
            # Create x data based on the length of the data
            x_data = np.arange(len(error_history))
            
            # Update the lines
            self.error_line.set_data(x_data, error_history)
            self.left_motor_line.set_data(x_data, left_speed_history)
            self.right_motor_line.set_data(x_data, right_speed_history)
            self.p_line.set_data(x_data, proportional_history)
            self.i_line.set_data(x_data, integral_history)
            self.d_line.set_data(x_data, derivative_history)
            self.diff_line.set_data(x_data, motor_diff_history)
            
            # Adjust axis limits if needed
            for ax in [self.error_ax, self.motor_ax, self.pid_ax, self.diff_ax]:
                ax.relim()
                ax.autoscale_view()
                
            # Draw the canvas
            self.canvas.draw_idle()
        except Exception as e:
            print(f"Error updating plots: {e}")
    
    def on_closing(self):
        """Handle window close event."""
        self.is_closed = True
        
        # Cancel any pending after() calls
        if self.check_queue_id is not None:
            try:
                self.root.after_cancel(self.check_queue_id)
            except:
                pass
        
        try:
            self.root.quit()
            self.root.destroy()
        except:
            pass 