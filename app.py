import cv2
import numpy as np
import mediapipe as mp
import os
import time
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import speech_recognition as sr
import threading
import queue
import glob
import tempfile
import subprocess


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class PresentationUploader:
    def __init__(self, width=1280, height=720):
        self.width = width
        self.height = height
        self.slides = []
        self.slide_paths = []
        self.temp_dir = None

    def create_upload_window(self):
        """Create a window for uploading presentation slides"""
        self.root = tk.Tk()
        self.root.title("Presentation Uploader")
        self.root.geometry("800x600")

        # Create a frame for options
        option_frame = tk.Frame(self.root)
        option_frame.pack(pady=10)

        # Option 1: Use sample slides
        sample_btn = tk.Button(option_frame, text="Use Sample Slides",
                               command=self.use_sample_slides, height=2, width=20)
        sample_btn.pack(side=tk.LEFT, padx=10)

        # Option 2: Select from files
        upload_btn = tk.Button(option_frame, text="Upload Image Files",
                               command=self.upload_image_files, height=2, width=20)
        upload_btn.pack(side=tk.LEFT, padx=10)

        # Option 3: Select PPT/PDF
        ppt_btn = tk.Button(option_frame, text="Upload PowerPoint/PDF",
                            command=self.upload_presentation, height=2, width=20)
        ppt_btn.pack(side=tk.LEFT, padx=10)

        # Option 4: Use webcam directly
        webcam_btn = tk.Button(option_frame, text="Skip to Webcam",
                               command=self.skip_to_webcam, height=2, width=20)
        webcam_btn.pack(side=tk.LEFT, padx=10)

        # Create a frame for slide preview
        preview_container = tk.Frame(self.root)
        preview_container.pack(pady=10, fill=tk.BOTH, expand=True)

        # Create a label for preview title
        preview_label = tk.Label(preview_container, text="Slide Preview:", font=("Arial", 12, "bold"))
        preview_label.pack(anchor=tk.W, padx=10)

        # Create a frame for slide preview
        self.preview_frame = tk.Frame(preview_container)
        self.preview_frame.pack(pady=5, fill=tk.BOTH, expand=True)

        # Progress bar for loading slides
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.root, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, padx=20, pady=5)

        # Instructions
        instructions = tk.Label(self.root, text="Select presentation slides or use sample slides.",
                                font=("Arial", 12))
        instructions.pack(pady=10)

        # Status label
        self.status_label = tk.Label(self.root, text="", font=("Arial", 10))
        self.status_label.pack(pady=5)

        # Start button (initially disabled)
        self.start_btn = tk.Button(self.root, text="Start Presentation",
                                   command=self.start_presentation, state=tk.DISABLED,
                                   height=2, width=20)
        self.start_btn.pack(pady=10)

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()

        return self.slides

    def use_sample_slides(self):
        """Use sample slides"""
        self.slides = []
        self.slide_paths = []
        self.clear_preview()

        # Create sample slides
        for i in range(5):
            slide = np.ones((self.height, self.width, 3), np.uint8) * 255
            cv2.putText(slide, f"Sample Slide {i + 1}", (self.width // 2 - 150, self.height // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
            self.slides.append(slide)

        self.display_slides_preview()
        self.status_label.config(text="Sample slides loaded successfully.")
        self.start_btn.config(state=tk.NORMAL)

    def upload_image_files(self):
        """Upload image files for presentation"""
        file_paths = filedialog.askopenfilenames(
            title="Select Presentation Slides",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )

        if not file_paths:
            return

        self.slides = []
        self.slide_paths = list(file_paths)
        self.clear_preview()

        # Load and process each image
        total_files = len(self.slide_paths)
        for i, path in enumerate(self.slide_paths):
            try:
                # Update progress
                self.progress_var.set((i / total_files) * 100)
                self.root.update()

                # Read the image
                img = cv2.imread(path)
                if img is None:
                    messagebox.showerror("Error", f"Failed to load image: {path}")
                    continue

                # Resize to fit the presentation
                img = cv2.resize(img, (self.width, self.height))
                self.slides.append(img)
            except Exception as e:
                messagebox.showerror("Error", f"Error processing image {path}: {str(e)}")

        self.progress_var.set(100)

        if self.slides:
            self.display_slides_preview()
            self.status_label.config(text=f"{len(self.slides)} slides loaded successfully.")
            self.start_btn.config(state=tk.NORMAL)
        else:
            self.status_label.config(text="No valid slides were loaded.")
            self.start_btn.config(state=tk.DISABLED)

    def upload_presentation(self):
        """Upload PowerPoint or PDF presentation"""
        file_path = filedialog.askopenfilename(
            title="Select Presentation File",
            filetypes=[("Presentation files", "*.ppt *.pptx *.pdf")]
        )

        if not file_path:
            return

        # Clear previous slides
        self.slides = []
        self.slide_paths = []
        self.clear_preview()

        # Create a temporary directory to store exported slides
        if self.temp_dir:
            self.cleanup_temp_files()

        self.temp_dir = tempfile.mkdtemp()

        try:
            file_ext = os.path.splitext(file_path)[1].lower()

            if file_ext in ['.ppt', '.pptx']:
                self.status_label.config(text="Converting PowerPoint to images...")
                self.convert_ppt_to_images(file_path)
            elif file_ext == '.pdf':
                self.status_label.config(text="Converting PDF to images...")
                self.convert_pdf_to_images(file_path)

            # Load the generated images
            self.load_slides_from_temp_dir()

            if self.slides:
                self.display_slides_preview()
                self.status_label.config(text=f"{len(self.slides)} slides loaded successfully.")
                self.start_btn.config(state=tk.NORMAL)
            else:
                self.status_label.config(text="No valid slides were extracted.")
                self.start_btn.config(state=tk.DISABLED)

        except Exception as e:
            messagebox.showerror("Error", f"Error processing presentation: {str(e)}")
            self.status_label.config(text="Failed to process presentation.")

    def convert_ppt_to_images(self, ppt_path):
        """Convert PowerPoint to images on Linux/Ubuntu with improved error handling"""
        try:
            # For Linux systems, use LibreOffice
            if self.is_command_available("libreoffice"):
                self.status_label.config(text="Converting using LibreOffice...")

                # Create output directory if it doesn't exist
                os.makedirs(self.temp_dir, exist_ok=True)

                # Get the base filename without extension
                base_filename = os.path.splitext(os.path.basename(ppt_path))[0]

                # Try direct conversion to PNG first
                self.status_label.config(text="Attempting direct PNG conversion...")
                cmd_to_png = ["libreoffice", "--headless", "--convert-to", "png",
                              "--outdir", self.temp_dir, ppt_path]

                try:
                    # Run the conversion process with full output capture
                    process = subprocess.run(cmd_to_png,
                                             stdout=subprocess.PIPE,
                                             stderr=subprocess.PIPE,
                                             text=True,
                                             timeout=60)  # Add timeout

                    print(f"LibreOffice stdout: {process.stdout}")
                    print(f"LibreOffice stderr: {process.stderr}")

                    # Check if PNG files were created
                    png_files = glob.glob(os.path.join(self.temp_dir, "*.png"))
                    if png_files:
                        self.status_label.config(text="PNG conversion successful.")
                        return
                    else:
                        self.status_label.config(text="PNG conversion failed, trying PDF method...")
                except subprocess.TimeoutExpired:
                    self.status_label.config(text="PNG conversion timed out, trying PDF method...")
                except Exception as e:
                    self.status_label.config(text=f"PNG conversion error: {str(e)}")

                # If direct PNG conversion failed, try PDF conversion
                pdf_output = os.path.join(self.temp_dir, f"{base_filename}.pdf")
                cmd_to_pdf = ["libreoffice", "--headless", "--convert-to", "pdf",
                              "--outdir", self.temp_dir, ppt_path]

                try:
                    # Run the PDF conversion process
                    process = subprocess.run(cmd_to_pdf,
                                             stdout=subprocess.PIPE,
                                             stderr=subprocess.PIPE,
                                             text=True,
                                             timeout=60)  # Add timeout

                    print(f"PDF conversion stdout: {process.stdout}")
                    print(f"PDF conversion stderr: {process.stderr}")

                    # Check if PDF was created
                    if os.path.exists(pdf_output):
                        self.status_label.config(text="PDF created, converting to images...")
                        self.convert_pdf_to_images(pdf_output)
                    else:
                        available_files = os.listdir(self.temp_dir)
                        error_msg = f"PDF conversion failed. Available files: {available_files}"
                        raise Exception(error_msg)
                except Exception as e:
                    raise Exception(f"PDF conversion error: {str(e)}")

            elif self.is_command_available("unoconv"):
                # Alternative: Try unoconv if available
                self.status_label.config(text="Converting using unoconv...")

                # Convert to PDF
                pdf_output = os.path.join(self.temp_dir, "presentation.pdf")
                cmd = ["unoconv", "-f", "pdf", "-o", pdf_output, ppt_path]

                try:
                    process = subprocess.run(cmd,
                                             stdout=subprocess.PIPE,
                                             stderr=subprocess.PIPE,
                                             text=True,
                                             timeout=60)

                    print(f"unoconv stdout: {process.stdout}")
                    print(f"unoconv stderr: {process.stderr}")

                    # Convert PDF to images
                    if os.path.exists(pdf_output):
                        self.convert_pdf_to_images(pdf_output)
                    else:
                        raise Exception("PDF conversion failed")
                except Exception as e:
                    raise Exception(f"unoconv error: {str(e)}")

            else:
                # If neither is available, suggest installation
                raise Exception("Neither LibreOffice nor unoconv is installed. Please install them with:\n"
                                "sudo apt-get install libreoffice\n"
                                "or\n"
                                "sudo apt-get install unoconv")

        except Exception as e:
            # Log the error for debugging
            print(f"PowerPoint conversion failed: {str(e)}")
            raise Exception(f"PowerPoint conversion failed: {str(e)}")

    def convert_pdf_to_images(self, pdf_path):
        """Convert PDF to images using pdf2image instead of PyMuPDF"""
        try:
            from pdf2image import convert_from_path

            # Convert PDF to images
            images = convert_from_path(pdf_path, dpi=200)

            # Save images
            for i, image in enumerate(images):
                # Update progress
                self.progress_var.set(((i + 1) / len(images)) * 100)
                self.root.update()

                # Save the image
                image_path = os.path.join(self.temp_dir, f"slide_{i + 1:03d}.png")
                image.save(image_path, "PNG")

                # Create an OpenCV image from the PIL image
                img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

                # Resize to fit the presentation
                img_cv = cv2.resize(img_cv, (self.width, self.height))

                # Add to slides
                self.slides.append(img_cv)
                self.slide_paths.append(image_path)

            self.progress_var.set(100)

        except Exception as e:
            raise Exception(f"PDF to image conversion failed: {str(e)}")

    def load_slides_from_temp_dir(self):
        """Load slides from the temporary directory"""
        image_files = glob.glob(os.path.join(self.temp_dir, "*.png"))
        image_files.sort()

        total_files = len(image_files)
        for i, path in enumerate(image_files):
            try:
                # Update progress
                self.progress_var.set((i / total_files) * 100)
                self.root.update()

                # Read the image
                img = cv2.imread(path)
                if img is None:
                    continue

                # Resize to fit the presentation
                img = cv2.resize(img, (self.width, self.height))
                self.slides.append(img)
                self.slide_paths.append(path)
            except Exception as e:
                print(f"Error loading image {path}: {str(e)}")

        self.progress_var.set(100)

    def is_command_available(self, command):
        """Check if a command is available on the system"""
        try:
            subprocess.run([command, "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except:
            return False

    def skip_to_webcam(self):
        """Skip slide selection and go directly to webcam"""
        self.slides = []
        self.use_sample_slides()  # Use sample slides as a fallback
        self.start_presentation()

    def clear_preview(self):
        """Clear the slide preview area"""
        for widget in self.preview_frame.winfo_children():
            widget.destroy()
    def display_slides_preview(self):
        """Display thumbnails of the loaded slides"""
        if not self.slides:
            return

        # Create a canvas for displaying thumbnails
        canvas = tk.Canvas(self.preview_frame)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Create a scrollbar
        scrollbar = tk.Scrollbar(self.preview_frame, orient="horizontal", command=canvas.xview)
        scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        canvas.configure(xscrollcommand=scrollbar.set)

        # Create a frame inside the canvas to hold the thumbnails
        thumbnail_frame = tk.Frame(canvas)
        canvas.create_window((0, 0), window=thumbnail_frame, anchor="nw")

        # Create thumbnails for each slide
        for i, slide in enumerate(self.slides):
            # Convert the slide to a format suitable for tkinter
            slide_rgb = cv2.cvtColor(slide, cv2.COLOR_BGR2RGB)
            slide_pil = Image.fromarray(slide_rgb)

            # Create a thumbnail
            thumbnail_width = 160
            thumbnail_height = 90
            slide_pil.thumbnail((thumbnail_width, thumbnail_height))

            # Convert to PhotoImage
            slide_photo = ImageTk.PhotoImage(slide_pil)

            # Create a label for the thumbnail
            thumbnail_label = tk.Label(thumbnail_frame, image=slide_photo)
            thumbnail_label.image = slide_photo  # Keep a reference
            thumbnail_label.pack(side=tk.LEFT, padx=5, pady=5)

            # Add slide number
            slide_num_label = tk.Label(thumbnail_frame, text=f"Slide {i + 1}")
            slide_num_label.place(x=thumbnail_label.winfo_x() + 5, y=thumbnail_label.winfo_y() + 5)

        # Update the canvas to show scrollbars if needed
        thumbnail_frame.update_idletasks()
        canvas.config(scrollregion=canvas.bbox("all"))

    def start_presentation(self):
        """Start the presentation with the loaded slides"""
        if not self.slides:
            messagebox.showerror("Error", "No slides available. Please load slides first.")
            return

        self.root.quit()
        self.root.destroy()

    def cleanup_temp_files(self):
        """Clean up temporary files"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            for file in os.listdir(self.temp_dir):
                try:
                    os.remove(os.path.join(self.temp_dir, file))
                except:
                    pass
            try:
                os.rmdir(self.temp_dir)
            except:
                pass

    def on_closing(self):
        """Handle window closing event"""
        self.cleanup_temp_files()
        self.root.destroy()
        sys.exit(0)

class VoiceCommandHandler:
    """Handles voice recognition and command processing"""
    def __init__(self, command_callback):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.command_callback = command_callback
        self.is_listening = False
        self.command_queue = queue.Queue()
        self.listen_thread = None
        
        # Available voice commands
        self.commands = {
            "next slide": "next_slide",
            "previous slide": "previous_slide",
            "go back": "previous_slide",
            "pointer": "pointer",
            "draw": "draw",
            "drawing": "draw",
            "erase": "erase",
            "clear": "erase",
            "zoom in": "zoom_in",
            "zoom out": "zoom_out",
            "reset zoom": "reset_zoom",
            "start timer": "start_timer",
            "stop timer": "stop_timer",
            "red pointer": "red_pointer",
            "green pointer": "green_pointer",
            "blue pointer": "blue_pointer",
            "first slide": "first_slide",
            "last slide": "last_slide",
            "stop listening": "stop_listening"
        }
        
        # Adjust recognizer sensitivity
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
    
    def start_listening(self):
        """Start listening for voice commands in a separate thread"""
        if self.listen_thread is not None and self.listen_thread.is_alive():
            return
            
        self.is_listening = True
        self.listen_thread = threading.Thread(target=self._listen_loop)
        self.listen_thread.daemon = True
        self.listen_thread.start()
        print("Voice command system activated. Listening for commands...")
    
    def stop_listening(self):
        """Stop listening for voice commands"""
        self.is_listening = False
        if self.listen_thread is not None:
            self.listen_thread.join(timeout=1)
        print("Voice command system deactivated")
    
    def _listen_loop(self):
        """Background thread that listens for voice commands"""
        while self.is_listening:
            try:
                with self.microphone as source:
                    print("Listening...")
                    audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=3)
                
                try:
                    text = self.recognizer.recognize_google(audio).lower()
                    print(f"Recognized: {text}")
                    
                    # Check if the text matches any command
                    for command_text, command_action in self.commands.items():
                        if command_text in text:
                            self.command_queue.put(command_action)
                            print(f"Command detected: {command_action}")
                            break
                except sr.UnknownValueError:
                    # Speech was unintelligible
                    pass
                except sr.RequestError as e:
                    print(f"Could not request results; {e}")
            except Exception as e:
                print(f"Error in voice recognition: {e}")
    
    def get_command(self):
        """Get the next command from the queue if available"""
        try:
            return self.command_queue.get_nowait()
        except queue.Empty:
            return None

class HandGesturePresentation:
    def __init__(self, slides=None):
        # Presentation parameters - define these first
        self.width, self.height = 1280, 720

        # Initialize MediaPipe hands module
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False,
                                         max_num_hands=1,
                                         min_detection_confidence=0.5,
                                         min_tracking_confidence=0.5)
        self.mp_draw = mp.solutions.drawing_utils

        # Pointer trail effect variables
        self.pointer_trail = []
        self.max_trail_length = 20
        self.pointer_color = (0, 255, 0)  # Green by default

        # Zoom functionality variables
        self.zoom_mode = False
        self.zoom_mode = False
        self.zoom_level = 1.0
        self.zoom_center = (self.width // 2, self.height // 2)
        self.zoom_max = 3.0
        self.zoom_min = 1.0
        self.zoom_step = 0.1

        # Presentation timer variables
        self.presentation_start_time = None
        self.timer_visible = True
        self.timer_duration = 0  # 0 means count up, otherwise count down from this value (in seconds)

        # Drawing parameters
        self.draw_color = ( 0, 0,255)  # Red color for drawing
        self.brush_thickness = 15
        self.eraser_thickness = 50

        # Presentation parameters

        self.canvas = np.zeros((self.height, self.width, 3), np.uint8)
        self.pointer_mode = False
        self.drawing_mode = False
        self.previous_time = 0

        # Load slides
        if slides is not None and len(slides) > 0:
            self.slides = slides
        else:
            self.slides = self.load_sample_slides()
        print(f"Total slides loaded: {len(self.slides)}")  # Debug output
        self.current_slide = 0

        self.current_slide = 0

        # Voice command system - Choose only one approach:
        # Option 1: Using callback
        self.voice_handler = VoiceCommandHandler(self.handle_voice_command)
        
        # Option 2: Using queue
        # self.voice_handler = VoiceCommandHandler()
        
        self.voice_control_enabled = False
        # Gesture cooldown to prevent multiple actions
        self.last_gesture_time = time.time()
        self.cooldown = 2.5 # 1 second cooldown between gestures

        # Mode tracking
        self.active_mode = "navigation"  # Can be: navigation, pointer, drawin

        # Drawing track points
        self.prev_x, self.prev_y = 0, 0

        
    
        # Debug flag to print recognized gestures
        self.debug = True

    def load_sample_slides(self):
        """Load sample presentation slides or create blank ones"""
        slides = []
        for i in range(5):
            slide = np.ones((self.height, self.width, 3), np.uint8) * 255
            cv2.putText(slide, f"Sample Slide {i + 1}", (self.width // 2 - 150, self.height // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
            slides.append(slide)
        return slides

    def find_hand_landmarks(self, frame):
        """Detect hand landmarks using MediaPipe"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        landmarks = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                for lm in hand_landmarks.landmark:
                    cx, cy = int(lm.x * self.width), int(lm.y * self.height)
                    landmarks.append((cx, cy))

        return landmarks
    
    def handle_voice_command(self, command):
        """Handle a voice command"""
        print(f"Handling voice command: {command}")
        current_time = time.time()
        
        # Handle the command based on its type
        if command == "next_slide":
            if self.current_slide < len(self.slides) - 1:
                self.current_slide += 1
                self.canvas = np.zeros((self.height, self.width, 3), np.uint8)
                print(f"Voice command: Next slide - Now showing slide {self.current_slide + 1}/{len(self.slides)}")
        
        elif command == "previous_slide":
            if self.current_slide > 0:
                self.current_slide -= 1
                self.canvas = np.zeros((self.height, self.width, 3), np.uint8)
                print(f"Voice command: Previous slide - Now showing slide {self.current_slide + 1}/{len(self.slides)}")

    
    def toggle_voice_control(self):
        """Toggle voice control on/off"""
        if self.voice_control_enabled:
            self.voice_handler.stop_listening()
            self.voice_control_enabled = False
        else:
            self.voice_handler.start_listening()
            self.voice_control_enabled = True
        print(f"Voice control {'enabled' if self.voice_control_enabled else 'disabled'}")
    
    def process_voice_commands(self):
        """Process any pending voice commands"""
        if not self.voice_control_enabled:
            return
            
        command = self.voice_handler.get_command()
        if command:
            current_time = time.time()
            
            # Handle the command based on its type
            if command == "next_slide":
                if self.current_slide < len(self.slides) - 1:
                    self.current_slide += 1
                    self.canvas = np.zeros((self.height, self.width, 3), np.uint8)
                    print(f"Voice command: Next slide - Now showing slide {self.current_slide + 1}/{len(self.slides)}")
            
            elif command == "previous_slide":
                if self.current_slide > 0:
                    self.current_slide -= 1
                    self.canvas = np.zeros((self.height, self.width, 3), np.uint8)
                    print(f"Voice command: Previous slide - Now showing slide {self.current_slide + 1}/{len(self.slides)}")
            
            elif command == "pointer":
                self.active_mode = "pointer"
                print("Voice command: Pointer mode activated")
            
            elif command == "draw":
                self.active_mode = "drawing"
                print("Voice command: Drawing mode activated")
            
            elif command == "erase":
                self.canvas = np.zeros((self.height, self.width, 3), np.uint8)
                self.active_mode = "navigation"
                print("Voice command: Canvas cleared")
            
            elif command == "zoom_in":
                self.zoom_mode = True
                self.zoom_level = min(self.zoom_level + self.zoom_step, self.zoom_max)
                print(f"Voice command: Zooming in - Level: {self.zoom_level:.1f}")
            
            elif command == "zoom_out":
                self.zoom_level = max(self.zoom_level - self.zoom_step, self.zoom_min)
                if self.zoom_level <= 1.0:
                    self.zoom_mode = False
                print(f"Voice command: Zooming out - Level: {self.zoom_level:.1f}")
            
            elif command == "reset_zoom":
                self.zoom_mode = False
                self.zoom_level = 1.0
                print("Voice command: Zoom reset")
            
            elif command == "start_timer":
                self.start_presentation_timer()
                print("Voice command: Timer started")
            
            elif command == "stop_timer":
                self.timer_visible = not self.timer_visible
                print(f"Voice command: Timer {'hidden' if not self.timer_visible else 'shown'}")
            
            elif command == "red_pointer":
                self.pointer_color = (0, 0, 255)  # BGR format
                print("Voice command: Pointer color changed to red")
            
            elif command == "green_pointer":
                self.pointer_color = (0, 255, 0)  # BGR format
                print("Voice command: Pointer color changed to green")
            
            elif command == "blue_pointer":
                self.pointer_color = (255, 0, 0)  # BGR format
                print("Voice command: Pointer color changed to blue")
            
            elif command == "first_slide":
                self.current_slide = 0
                self.canvas = np.zeros((self.height, self.width, 3), np.uint8)
                print("Voice command: First slide")
            
            elif command == "last_slide":
                self.current_slide = len(self.slides) - 1
                self.canvas = np.zeros((self.height, self.width, 3), np.uint8)
                print("Voice command: Last slide")
            
            elif command == "stop_listening":
                self.toggle_voice_control()
                print("Voice command: Voice control disabled")
    
    def display_voice_control_status(self, frame):
        """Display voice control status on frame"""
        status = "Voice Control: ON" if self.voice_control_enabled else "Voice Control: OFF"
        color = (0, 255, 0) if self.voice_control_enabled else (0, 0, 255)  # Green if enabled, red if disabled
        cv2.putText(frame, status, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


    def recognize_gesture(self, landmarks):
        """Recognize hand gestures based on finger positions"""
        if not landmarks or len(landmarks) < 21:
            return None

        # Extract fingertip positions
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]

        # Extract finger MCP positions (knuckles)
        thumb_mcp = landmarks[2]
        index_mcp = landmarks[5]
        middle_mcp = landmarks[9]
        ring_mcp = landmarks[13]
        pinky_mcp = landmarks[17]

        # Extract wrist position for better orientation detection
        wrist = landmarks[0]

        # Calculate if fingers are extended by comparing y-coordinates
        # (lower y value means finger is higher/extended since origin is top-left)

        # For thumb, we need to check if it's to the left/right of the MCP based on hand orientation
        # This is a simplified approach - for more accurate detection, consider hand handedness
        is_right_hand = thumb_mcp[0] < wrist[0]  # Simple check for hand orientation

        if is_right_hand:
            thumb_up = thumb_tip[0] < thumb_mcp[0]  # Thumb is extended left for right hand
        else:
            thumb_up = thumb_tip[0] > thumb_mcp[0]  # Thumb is extended right for left hand

        # Check if fingers are up (finger tip is above finger MCP)
        index_up = index_tip[1] < index_mcp[1] - 15  # Adding threshold for more reliable detection
        middle_up = middle_tip[1] < middle_mcp[1] - 15
        ring_up = ring_tip[1] < ring_mcp[1] - 15
        pinky_up = pinky_tip[1] < pinky_mcp[1] - 15

        # # Add debugging information on frame
        # if self.debug:
        #     fingers_state = f"T:{thumb_up} I:{index_up} M:{middle_up} R:{ring_up} P:{pinky_up}"
        #     print(fingers_state)

        # Recognize gestures as per the research paper
        # Gesture 1: Thumb Finger - Move to Previous Slide
        # Gesture 1: Thumb Finger - Previous Slide (more lenient)
        if thumb_up and not any([index_up, middle_up, ring_up, pinky_up]):
            return "previous_slide"

        # Gesture 2: Pinky Finger - Next Slide (more lenient)
        elif pinky_up and not any([thumb_up, index_up, middle_up, ring_up]):
            return "next_slide"

        # Gesture 3: Index Finger and Middle Finger Together - Holding the Pointer
        elif not thumb_up and index_up and middle_up and not ring_up and not pinky_up:
            return "pointer"

        # Gesture 4: Index Finger - Drawing on the Slide
        elif not thumb_up and index_up and not middle_up and not ring_up and not pinky_up:
            return "draw"

        # Gesture 5: Middle Three Fingers - Erase/Undo the Previous Draw
        elif not thumb_up and not index_up and middle_up and ring_up and pinky_up:
            return "erase"

        # Gesture 6: Four fingers up (index through pinky) - Toggle zoom mode
        elif not thumb_up and index_up and middle_up and ring_up and pinky_up:
            return "toggle_zoom"
        
        # Add a new gesture for toggling voice control (thumb and pinky up)
        elif thumb_up and not index_up and not middle_up and not ring_up and pinky_up:
            return "toggle_voice"

        else:
            return None


    def handle_gesture(self, gesture, landmarks):
        """Handle recognized gestures with mode management"""
        current_time = time.time()
        
        # Cooldown check first
        if current_time - self.last_gesture_time < self.cooldown:
            return

        # Handle mode transitions
        if gesture == "previous_slide":
            self.active_mode = "navigation"
            if self.current_slide > 0:
                self.current_slide -= 1
                self.canvas = np.zeros((self.height, self.width, 3), np.uint8)
                print(f"Previous slide - Now showing slide {self.current_slide + 1}/{len(self.slides)}")
            self.last_gesture_time = current_time
        
        elif gesture == "next_slide":
            self.active_mode = "navigation"
            if self.current_slide < len(self.slides) - 1:
                self.current_slide += 1
                self.canvas = np.zeros((self.height, self.width, 3), np.uint8)
                print(f"Next slide - Now showing slide {self.current_slide + 1}/{len(self.slides)}")
            self.last_gesture_time = current_time
        
        elif gesture == "pointer":
            self.active_mode = "pointer"
            self.last_gesture_time = current_time
            print("Pointer mode activated")
        
        elif gesture == "draw":
            self.active_mode = "drawing"
            self.last_gesture_time = current_time
            print("Drawing mode activated")
        
        elif gesture == "erase":
            self.canvas = np.zeros((self.height, self.width, 3), np.uint8)
            self.active_mode = "navigation"  # New line added here
            self.last_gesture_time = current_time
            print("Canvas cleared and drawing mode exited")
        
        # Add handling for the zoom control toggle gesture
        elif gesture == "toggle_zoom":
            self.toggle_zoom_mode()
            self.last_gesture_time = current_time
            print("Zoom mode toggled")

        # Add handling for the voice control toggle gesture
        elif gesture == "toggle_voice":
            self.toggle_voice_control()
            self.last_gesture_time = current_time
            print("Voice control toggled")

       

    def draw_pointer(self, frame, landmarks):
        """Draw pointer with trail effect"""
        if self.active_mode == "pointer" and landmarks and len(landmarks) >= 9:
            # Update trail first
            self.update_pointer_trail(landmarks)

            # Draw fading trail
            for i, (x, y) in enumerate(self.pointer_trail):
                # Calculate alpha (opacity) based on position in trail
                alpha = int(255 * (i / len(self.pointer_trail)))
                # Create color with fading effect
                color = list(self.pointer_color)
                # Draw circle with size based on position in trail
                radius = 5 + int(10 * (i / len(self.pointer_trail)))
                cv2.circle(frame, (x, y), radius, color, cv2.FILLED)

            # Draw current pointer position (index fingertip)
            cx, cy = landmarks[8]
            cv2.circle(frame, (cx, cy), 15, self.pointer_color, cv2.FILLED)

    def update_pointer_trail(self, landmarks):
        """Update the pointer trail positions"""
        if self.active_mode == "pointer" and landmarks and len(landmarks) >= 9:
            cx, cy = landmarks[8]  # Index finger tip
            self.pointer_trail.append((cx, cy))
            # Keep trail at max length
            if len(self.pointer_trail) > self.max_trail_length:
                self.pointer_trail.pop(0)

    def draw_on_canvas(self, landmarks):
        """Draw only in drawing mode"""
        if self.active_mode == "drawing" and landmarks and len(landmarks) >= 9:
            cx, cy = landmarks[8]
            if self.prev_x == 0 and self.prev_y == 0:
                self.prev_x, self.prev_y = cx, cy
            else:
                cv2.line(self.canvas, (self.prev_x, self.prev_y), (cx, cy),
                         self.draw_color, self.brush_thickness)
            self.prev_x, self.prev_y = cx, cy
        else:
            self.prev_x, self.prev_y = 0, 0

    def toggle_zoom_mode(self):
        """Toggle zoom mode on/off"""
        self.zoom_mode = not self.zoom_mode
        if not self.zoom_mode:
            # Reset zoom when exiting zoom mode
            self.zoom_level = 1.0
            self.zoom_center = (self.width // 2, self.height // 2)
        print(f"Zoom mode {'activated' if self.zoom_mode else 'deactivated'}")

    def handle_zoom(self, landmarks):
        """Handle zoom in/out based on hand gesture"""
        if not self.zoom_mode or not landmarks or len(landmarks) < 21:
            return

        # Get thumb and index finger positions
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]

        # Calculate distance between thumb and index finger
        distance = np.sqrt((thumb_tip[0] - index_tip[0]) ** 2 + (thumb_tip[1] - index_tip[1]) ** 2)

        # Use distance to control zoom level (simplified approach)
        # This is a basic implementation - you might want a more sophisticated gesture
        # Normalize distance to a reasonable range for zoom control
        normalized_distance = distance / 100

        # Update zoom level with some damping to prevent jumps
        target_zoom = np.clip(normalized_distance, self.zoom_min, self.zoom_max)
        self.zoom_level = self.zoom_level * 0.9 + target_zoom * 0.1

        # Update zoom center based on hand position (middle of palm)
        palm_center = landmarks[0]  # Using wrist as approximate palm center
        self.zoom_center = palm_center

    def apply_zoom(self, slide):
        """Apply zoom effect to the slide"""
        if self.zoom_level <= 1.0:
            return slide

        # Calculate the region to zoom into
        zoom_width = int(self.width / self.zoom_level)
        zoom_height = int(self.height / self.zoom_level)

        # Calculate top-left corner of zoom region, ensuring it stays within bounds
        x1 = max(0, min(self.width - zoom_width, self.zoom_center[0] - zoom_width // 2))
        y1 = max(0, min(self.height - zoom_height, self.zoom_center[1] - zoom_height // 2))

        # Extract region and resize to full frame
        zoom_region = slide[y1:y1 + zoom_height, x1:x1 + zoom_width]
        zoomed_slide = cv2.resize(zoom_region, (self.width, self.height))

        return zoomed_slide

    def display_fps(self, frame):
        """Display the FPS on the frame"""
        current_time = time.time()
        fps = 1 / (current_time - self.previous_time)
        self.previous_time = current_time
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    def display_current_slide_info(self, frame):
        """Updated mode display"""
        mode_colors = {
            "navigation": (0, 255, 0),
            "pointer": (255, 0, 0),
            "drawing": (0, 0, 255)
        }
        color = mode_colors.get(self.active_mode, (0, 255, 0))
        cv2.putText(frame, f"Mode: {self.active_mode.capitalize()}", (10, 110), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    def start_presentation_timer(self):
        """Start or reset the presentation timer"""
        self.presentation_start_time = time.time()
        print("Presentation timer started")

    def display_timer(self, frame):
        """Display the presentation timer on the frame"""
        if not self.timer_visible or self.presentation_start_time is None:
            return

        current_time = time.time()
        elapsed_seconds = int(current_time - self.presentation_start_time)

        if self.timer_duration > 0:
            # Countdown timer
            remaining_seconds = max(0, self.timer_duration - elapsed_seconds)
            minutes, seconds = divmod(remaining_seconds, 60)
            timer_text = f"Time remaining: {minutes:02d}:{seconds:02d}"

            # Change color to red when less than 1 minute remaining
            if remaining_seconds < 60:
                color = (0, 0, 255)  # Red in BGR
            else:
                color = (255, 255, 255)  # White in BGR
        else:
            # Count-up timer
            minutes, seconds = divmod(elapsed_seconds, 60)
            timer_text = f"Time elapsed: {minutes:02d}:{seconds:02d}"
            color = (255, 255, 255)  # White in BGR

        # Display timer in the top right corner
        cv2.putText(frame, timer_text, (self.width - 300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


    def run(self):
        """Run the presentation with hand gesture control"""
        # Try different camera indices
        camera_index = 0
        max_attempts = 5  # Increase max attempts to try more camera indices
        cap = None

        # Start the presentation timer
        self.start_presentation_timer()

        # Process voice commands
        self.process_voice_commands()

        for _ in range(max_attempts):
            print(f"Trying to open camera at index {camera_index}")
            cap = cv2.VideoCapture(camera_index)
            if cap.isOpened():
                print(f"Successfully opened camera at index {camera_index}")
                # Set camera properties inside this scope where cap is valid
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                break
            cap.release()  # Make sure to release the camera if it failed to open
            camera_index += 1
            print(f"Failed to open camera at index {camera_index-1}")
        
        if not cap or not cap.isOpened():
            print("ERROR: Could not open any camera. Please check your webcam connection.")
            print("Continuing with presentation mode only (no webcam feed).")
            # Create a dummy frame to show instead of webcam feed
            dummy_frame = np.ones((self.height, self.width, 3), np.uint8) * 120
            cv2.putText(dummy_frame, "No webcam available", (self.width//4, self.height//2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
            
            # Just show the slides without webcam interaction
            while True:
                # Display the current slide
                cv2.imshow("Presentation", self.slides[self.current_slide])
                cv2.imshow("Webcam", dummy_frame)
                
                # Handle keyboard input instead of gestures
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('n') and self.current_slide < len(self.slides) - 1:
                    self.current_slide += 1
                elif key == ord('p') and self.current_slide > 0:
                    self.current_slide -= 1
            
            cv2.destroyAllWindows()
            return
        
        # If we get here, we have a working camera
        try:
            while True:
                # Read frame from webcam
                ret, frame = cap.read()
                if not ret:
                    print("Failed to read from camera")
                    break

                # Flip the frame horizontally for a more natural interaction
                frame = cv2.flip(frame, 1)

                # Resize frame to presentation dimensions
                frame = cv2.resize(frame, (self.width, self.height))

                # Find hand landmarks
                landmarks = self.find_hand_landmarks(frame)

                # Process voice commands - only needed if using the queue approach
                if self.voice_control_enabled:
                    self.process_voice_commands()

                # Recognize and handle gestures
                if landmarks:
                    gesture = self.recognize_gesture(landmarks)
                    if gesture:
                        self.handle_gesture(gesture, landmarks)

                    # Draw pointer or draw on canvas based on mode
                    self.draw_pointer(frame, landmarks)
                    self.draw_on_canvas(landmarks)
                else:
                    # Reset previous drawing points when hand not detected
                    self.prev_x, self.prev_y = 0, 0

                if landmarks:
                    gesture = self.recognize_gesture(landmarks)
                    if gesture:
                        self.handle_gesture(gesture, landmarks)

                    # Draw pointer or draw on canvas based on mode
                    self.draw_pointer(frame, landmarks)
                    self.draw_on_canvas(landmarks)

                    # Handle zoom if in zoom mode
                    if self.zoom_mode:
                        self.handle_zoom(landmarks)
                else:
                    # Reset previous drawing points when hand not detected
                    self.prev_x, self.prev_y = 0, 0



                # Get the current slide
                current_slide = self.slides[self.current_slide].copy()

                # Overlay the canvas on the slide
                slide_with_canvas = cv2.addWeighted(current_slide, 1, self.canvas, 1, 0)

                # Apply zoom effect if enabled
                if self.zoom_mode and self.zoom_level > 1.0:
                    slide_with_canvas = self.apply_zoom(slide_with_canvas)

                # Display the slide with canvas in a window
                cv2.imshow("Presentation", slide_with_canvas)

                # Add voice control status display
                self.display_voice_control_status(frame)

                # Display webcam feed with additional information
                self.display_fps(frame)
                self.display_current_slide_info(frame)
                self.display_timer(frame)  # Add timer display
                cv2.imshow("Webcam", frame)

                # Display the slide with canvas in a window
                cv2.imshow("Presentation", slide_with_canvas)

                # Exit on 'q' key press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            # Stop voice recognition when exiting
            if self.voice_control_enabled:
                self.voice_handler.stop_listening()

            # Release resources
            if cap is not None:
                cap.release()
            cv2.destroyAllWindows()
            self.hands.close()


def main():
    # Create uploader and get slides
    uploader = PresentationUploader()
    slides = uploader.create_upload_window()

    # Start the presentation
    presentation = HandGesturePresentation(slides)

    # Optionally, enable voice control at startup
    # presentation.toggle_voice_control()



    presentation.run()


if __name__ == "__main__":
    main()
