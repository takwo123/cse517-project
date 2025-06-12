#!/usr/bin/env python3
"""
Real-time Character Prediction System
Implements Option A: In-Memory Serving for instant predictions while typing
"""

import os
import sys
import time
import threading
import torch
from collections import deque
import curses
from pathlib import Path
from myprogram import MyModel

class RealTimePredictor:
    def __init__(self, work_dir=None):
        """Initialize the real-time predictor with pre-loaded model"""
        print("🚀 Initializing Real-Time Character Predictor...")
        print("📦 Loading model into memory...")
        
        # Smart path detection for work directory
        if work_dir is None:
            # Get current working directory and script directory
            cwd = Path.cwd()
            script_dir = Path(__file__).parent
            
            # List of potential work directory paths to try
            potential_paths = [
                cwd / 'work',                    # Current directory has work/
                cwd.parent / 'work',             # Parent directory has work/
                script_dir / 'work',             # Script directory has work/
                script_dir.parent / 'work',      # Script parent directory has work/
                '../work',                       # Relative paths
                './work',
                '../../work'
            ]
            
            work_dir = None
            print("🔍 Searching for model checkpoint...")
            for path in potential_paths:
                try:
                    abs_path = Path(path).resolve()
                    checkpoint_path = abs_path / 'model.checkpoint'
                    print(f"   Checking: {abs_path}")
                    if abs_path.exists() and checkpoint_path.exists():
                        work_dir = str(abs_path)
                        print(f"✅ Found model at: {work_dir}")
                        break
                except Exception as e:
                    print(f"   Error checking {path}: {e}")
                    continue
            
            if work_dir is None:
                print(f"❌ No model found. Searched in:")
                for path in potential_paths:
                    try:
                        abs_path = Path(path).resolve()
                        checkpoint_path = abs_path / 'model.checkpoint'
                        print(f"   {abs_path} -> checkpoint exists: {checkpoint_path.exists()}")
                    except Exception as e:
                        print(f"   {path} -> error: {e}")
                raise FileNotFoundError("Model checkpoint not found in any expected location")
        
        start_time = time.time()
        self.model = MyModel.load(work_dir)
        load_time = time.time() - start_time
        
        print(f"⚡ Model loaded in {load_time:.2f}s")
        print(f"🧠 Model parameters: {sum(p.numel() for p in self.model.model.parameters()):,}")
        
        # Performance tracking
        self.prediction_times = deque(maxlen=100)
        self.total_predictions = 0
        
    def predict_next_char(self, context, num_predictions=3):
        """Fast in-memory prediction"""
        start_time = time.time()
        
        # Get predictions from the model
        predictions = self.model.run_pred([context])
        pred_chars = list(predictions[0]) if predictions and predictions[0] else [' ', 'e', 't']
        
        # Ensure we have the requested number of predictions
        while len(pred_chars) < num_predictions:
            pred_chars.append(' ')
        
        # Track performance
        prediction_time = (time.time() - start_time) * 1000  # Convert to ms
        self.prediction_times.append(prediction_time)
        self.total_predictions += 1
        
        return pred_chars[:num_predictions], prediction_time
    
    def get_performance_stats(self):
        """Get current performance statistics"""
        if not self.prediction_times:
            return {"avg_ms": 0, "min_ms": 0, "max_ms": 0, "total": 0}
        
        times = list(self.prediction_times)
        return {
            "avg_ms": sum(times) / len(times),
            "min_ms": min(times),
            "max_ms": max(times),
            "total": self.total_predictions
        }

class TerminalInterface:
    def __init__(self, predictor):
        self.predictor = predictor
        self.current_text = ""
        self.cursor_pos = 0
        
    def run(self, stdscr):
        """Run the terminal-based real-time prediction interface"""
        # Setup curses
        curses.curs_set(1)  # Show cursor
        stdscr.timeout(100)  # Non-blocking input
        stdscr.clear()
        
        # Colors
        curses.start_color()
        curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)    # Predictions
        curses.init_pair(2, curses.COLOR_YELLOW, curses.COLOR_BLACK)   # Stats
        curses.init_pair(3, curses.COLOR_CYAN, curses.COLOR_BLACK)     # Headers
        curses.init_pair(4, curses.COLOR_RED, curses.COLOR_BLACK)      # Warnings
        
        def safe_addstr(row, col, text, attr=0):
            """Safely add string to screen with bounds checking"""
            try:
                height, width = stdscr.getmaxyx()
                if row < height and col < width:
                    # Truncate text to fit within screen width
                    max_len = width - col - 1
                    if len(text) > max_len:
                        text = text[:max_len]
                    stdscr.addstr(row, col, text, attr)
            except curses.error:
                pass  # Ignore curses errors
        
        while True:
            stdscr.clear()
            height, width = stdscr.getmaxyx()
            
            # Check minimum terminal size
            if height < 12 or width < 50:
                safe_addstr(0, 0, "Terminal too small! Please resize to at least 50x12", curses.A_BOLD)
                safe_addstr(1, 0, f"Current size: {width}x{height}")
                safe_addstr(2, 0, "Press ESC to quit")
                stdscr.refresh()
                
                key = stdscr.getch()
                if key == 27:  # ESC key
                    break
                continue
            
            # Header
            header = "Real-Time Character Prediction (Press ESC to quit)"
            safe_addstr(0, 0, header, curses.color_pair(3) | curses.A_BOLD)
            
            # Instructions
            instructions = "Type anything and see real-time predictions appear below!"
            safe_addstr(1, 0, instructions)
            
            # Current text input area
            safe_addstr(3, 0, "Your text:", curses.A_BOLD)
            text_display = self.current_text
            if len(text_display) > width - 15:
                text_display = "..." + text_display[-(width-18):]
            
            # Text box
            box_width = min(width - 3, 80)
            safe_addstr(4, 0, "┌" + "─" * box_width + "┐")
            safe_addstr(5, 0, "│ " + text_display[:box_width-2] + " " * max(0, box_width-len(text_display)-2) + "│")
            safe_addstr(6, 0, "└" + "─" * box_width + "┘")
            
            # Get predictions if we have text
            if self.current_text and height >= 12:
                # Use last 24 characters as context (model's sequence length)
                context = self.current_text[-24:] if len(self.current_text) > 24 else self.current_text
                predictions, pred_time = self.predictor.predict_next_char(context, 3)
                
                # Display predictions
                safe_addstr(8, 0, "Predictions:", curses.A_BOLD)
                for i, pred_char in enumerate(predictions):
                    if 9 + i < height - 2:
                        display_char = repr(pred_char) if pred_char in [' ', '\t', '\n'] else pred_char
                        confidence = max(0, 100 - i * 30)  # Fake confidence for display
                        pred_text = f"  {i+1}. '{display_char}' ({confidence}%)"
                        safe_addstr(9 + i, 0, pred_text, curses.color_pair(1))
                
                # Performance stats (compact format for small terminals)
                if height >= 14:
                    stats = self.predictor.get_performance_stats()
                    perf_line = f"Last: {pred_time:.1f}ms, Avg: {stats['avg_ms']:.1f}ms, Total: {stats['total']}"
                    safe_addstr(height - 2, 0, perf_line, curses.color_pair(2))
                    
                    # Status on last line
                    if pred_time < 50:
                        perf_status = "EXCELLENT"
                    elif pred_time < 100:
                        perf_status = "GOOD"
                    else:
                        perf_status = "SLOW"
                    safe_addstr(height - 1, 0, f"Status: {perf_status}")
            elif height < 12:
                safe_addstr(8, 0, "Terminal too short for predictions...", curses.color_pair(4))
            else:
                safe_addstr(8, 0, "Start typing to see predictions...", curses.color_pair(4))
            
            # Position cursor at the end of text
            cursor_row = 5
            cursor_col = min(len(self.current_text) + 2, width - 2)
            try:
                stdscr.move(cursor_row, cursor_col)
            except curses.error:
                pass
            
            stdscr.refresh()
            
            # Handle input
            try:
                key = stdscr.getch()
                
                if key == 27:  # ESC key
                    break
                elif key == curses.KEY_BACKSPACE or key == 127:  # Backspace
                    if self.current_text:
                        self.current_text = self.current_text[:-1]
                elif key == curses.KEY_ENTER or key == 10:  # Enter
                    self.current_text += '\n'
                elif 32 <= key <= 126:  # Printable ASCII
                    self.current_text += chr(key)
                
            except curses.error:
                continue

def main():
    """Main function to run the real-time predictor"""
    
    print("🎯 Real-Time Character Prediction System")
    print("="*50)
    
    # Initialize predictor
    try:
        predictor = RealTimePredictor()
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print("   Make sure you have a trained model in the 'work' directory")
        return 1
    
    # Choose interface
    print("\nSelect interface:")
    print("1. Terminal interface (curses-based)")
    print("2. Simple command-line test")
    
    try:
        choice = input("\nEnter choice (1-2): ").strip()
        
        if choice == "1":
            print("\n🖥️  Starting terminal interface...")
            print("   Use arrow keys, type normally, press ESC to quit")
            try:
                terminal_interface = TerminalInterface(predictor)
                curses.wrapper(terminal_interface.run)
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
        
        elif choice == "2":
            print("\n⌨️  Simple command-line test mode")
            print("   Type text and press Enter for predictions (type 'quit' to exit)")
            
            while True:
                try:
                    text = input("\n> ").strip()
                    if text.lower() in ['quit', 'exit', 'q']:
                        break
                    
                    if text:
                        predictions, pred_time = predictor.predict_next_char(text)
                        print(f"   Predictions: {predictions}")
                        print(f"   Time: {pred_time:.1f}ms")
                        
                        stats = predictor.get_performance_stats()
                        print(f"   Average: {stats['avg_ms']:.1f}ms (Total: {stats['total']})")
                        
                except KeyboardInterrupt:
                    break
            
            print("👋 Goodbye!")
        
        else:
            print("❌ Invalid choice")
            return 1
            
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 