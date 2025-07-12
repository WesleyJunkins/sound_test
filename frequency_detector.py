import pyaudio
import numpy as np
import threading
import time
from collections import deque

class FrequencyDetector:
    def __init__(self, sample_rate=44100, chunk_size=2048, buffer_size=3):
        """Initialize the frequency detector.
        
        Args:
            sample_rate: Audio sample rate (default 44100 Hz)
            chunk_size: Number of samples per chunk (default 2048 for better resolution)
            buffer_size: Number of frequency readings to average (default 3 for faster response)
        """
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.buffer_size = buffer_size
        self.frequency_buffer = deque(maxlen=buffer_size)
        self.is_running = False
        
        # Lower threshold for better sensitivity to sound.py tones
        self.magnitude_threshold = 0.005  # Reduced from 0.01
        
        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()
        
        # Open microphone stream
        self.stream = self.audio.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=sample_rate,
            input=True,
            frames_per_buffer=chunk_size
        )
    
    def detect_frequency(self, audio_data):
        """Detect the dominant frequency in the audio data using FFT."""
        # Apply window function to reduce spectral leakage
        windowed_data = audio_data * np.hanning(len(audio_data))
        
        # Perform FFT
        fft_data = np.fft.fft(windowed_data)
        fft_magnitude = np.abs(fft_data)
        
        # Get positive frequencies only (first half of FFT)
        positive_freqs = fft_magnitude[:len(fft_magnitude)//2]
        
        # Focus on frequency ranges used by sound.py (100-4000 Hz)
        # Calculate frequency resolution
        freq_resolution = self.sample_rate / len(fft_data)
        
        # Find the range of interest (100-4000 Hz)
        start_idx = int(100 / freq_resolution)
        end_idx = int(4000 / freq_resolution)
        
        # Limit search to relevant frequency range
        relevant_freqs = positive_freqs[start_idx:end_idx]
        
        if len(relevant_freqs) == 0:
            return 0, 0
        
        # Find the frequency with maximum magnitude in relevant range
        max_index = np.argmax(relevant_freqs)
        actual_index = start_idx + max_index
        
        # Convert index to frequency
        frequency = actual_index * freq_resolution
        
        # Calculate magnitude (normalized)
        magnitude = relevant_freqs[max_index] / len(fft_data)
        
        return frequency, magnitude
    
    def get_average_frequency(self):
        """Get the average frequency from the buffer."""
        if not self.frequency_buffer:
            return None, 0
        
        frequencies = [freq for freq, mag in self.frequency_buffer]
        magnitudes = [mag for freq, mag in self.frequency_buffer]
        
        avg_freq = np.mean(frequencies)
        avg_magnitude = np.mean(magnitudes)
        
        return avg_freq, avg_magnitude
    
    def listen(self):
        """Continuously listen for audio and detect frequencies."""
        print("Listening for audio from sound.py... Press Ctrl+C to stop.")
        print("Frequency (Hz) | Magnitude | Note | Status")
        print("-" * 50)
        
        last_freq = 0
        stable_count = 0
        
        try:
            while self.is_running:
                # Read audio data
                audio_data = np.frombuffer(
                    self.stream.read(self.chunk_size, exception_on_overflow=False),
                    dtype=np.float32
                )
                
                # Detect frequency
                frequency, magnitude = self.detect_frequency(audio_data)
                
                # More sensitive threshold for sound.py detection
                if magnitude > self.magnitude_threshold:
                    # Add to buffer
                    self.frequency_buffer.append((frequency, magnitude))
                    
                    # Get average frequency
                    avg_freq, avg_magnitude = self.get_average_frequency()
                    
                    if avg_freq is not None and avg_freq > 50:  # Filter out very low frequencies
                        # Convert frequency to musical note
                        note = self.frequency_to_note(avg_freq)
                        
                        # Check if frequency is stable (similar to last reading)
                        if abs(avg_freq - last_freq) < 10:  # Within 10 Hz
                            stable_count += 1
                            status = f"STABLE({stable_count})" if stable_count > 1 else "DETECTED"
                        else:
                            stable_count = 0
                            status = "DETECTED"
                        
                        last_freq = avg_freq
                        
                        # Print frequency information with more detail
                        print(f"{avg_freq:8.1f} Hz | {avg_magnitude:8.4f} | {note:>4} | {status}")
                
                # Faster response time for sound.py detection
                time.sleep(0.05)  # Reduced from 0.1
                
        except KeyboardInterrupt:
            print("\nStopping frequency detection...")
        finally:
            self.stop()
    
    def frequency_to_note(self, frequency):
        """Convert frequency to musical note name."""
        if frequency < 20:  # Below audible range
            return "---"
        
        # A4 = 440 Hz
        A4 = 440.0
        C0 = A4 * (2 ** (-4.75))  # C0 is 4.75 octaves below A4
        
        # Calculate semitones from C0
        semitones = 12 * np.log2(frequency / C0)
        
        # Round to nearest semitone
        semitone = round(semitones)
        
        # Note names
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
        # Calculate octave
        octave = (semitone // 12) - 1
        note_index = semitone % 12
        
        return f"{note_names[note_index]}{octave}"
    
    def start(self):
        """Start listening for frequencies."""
        self.is_running = True
        self.listen()
    
    def stop(self):
        """Stop listening and clean up resources."""
        self.is_running = False
        if hasattr(self, 'stream'):
            self.stream.stop_stream()
            self.stream.close()
        if hasattr(self, 'audio'):
            self.audio.terminate()

def main():
    """Main function to run the frequency detector."""
    print("Real-time Frequency Detector for sound.py")
    print("=" * 50)
    print("Optimized for detecting tones from sound.py program")
    print("Frequency range: 100-4000 Hz")
    print("Sensitivity: High")
    print()
    
    # Create and start frequency detector
    detector = FrequencyDetector()
    detector.start()

if __name__ == "__main__":
    main() 