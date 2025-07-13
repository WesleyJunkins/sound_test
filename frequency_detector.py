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
        
        # Define frequency signatures for sound.py comments
        self.frequency_signatures = {
            # Single frequencies
            430: "CRe",
            975: "CL/ACK",
            1650: "MS",
            1975: "CL",
            2145: "ANSam",
            2745: "INFO",
            2750: "INFO",
            1250: "INFO",
            1255: "INFO",
            
            # Frequency ranges (approximate)
            (350, 440): "Dial Tone",
            (697, 1336): "DTMF 2/5/8",
            (770, 1336): "DTMF 5/8",
            (770, 1477): "DTMF 6",
            (770, 1209): "DTMF 4",
            (852, 1336): "DTMF 8",
            (941, 1336): "DTMF 0",
            (697, 1477): "DTMF 3",
            
            # Random frequency ranges
            (950, 1300): "CL/ACK Random",
            (1600, 2000): "MS Random",
            (850, 1450): "ANSam Random",
            (2145, 2155): "ANSam Carrier",
            (750, 1750): "INFO Random",
            (1950, 2750): "INFO Random",
            (2745, 2755): "INFO Carrier",
            (1250, 1255): "INFO Carrier",
            
            # Special patterns
            (100, 3000): "Data",
            (1400, 2000): "Init",
            (1600, 2250): "Resp",
        }
        
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
    
    def detect_frequency_range(self, audio_data):
        """Detect multiple frequencies and frequency ranges in the audio data."""
        # Apply window function to reduce spectral leakage
        windowed_data = audio_data * np.hanning(len(audio_data))
        
        # Perform FFT
        fft_data = np.fft.fft(windowed_data)
        fft_magnitude = np.abs(fft_data)
        
        # Get positive frequencies only (first half of FFT)
        positive_freqs = fft_magnitude[:len(fft_magnitude)//2]
        
        # Calculate frequency resolution
        freq_resolution = self.sample_rate / len(fft_data)
        
        # Find the range of interest (100-4000 Hz)
        start_idx = int(100 / freq_resolution)
        end_idx = int(4000 / freq_resolution)
        
        # Limit search to relevant frequency range
        relevant_freqs = positive_freqs[start_idx:end_idx]
        
        if len(relevant_freqs) == 0:
            return [], 0
        
        # Find peaks above threshold
        threshold = np.max(relevant_freqs) * 0.3  # 30% of max magnitude
        peaks = []
        
        for i in range(1, len(relevant_freqs) - 1):
            if (relevant_freqs[i] > threshold and 
                relevant_freqs[i] > relevant_freqs[i-1] and 
                relevant_freqs[i] > relevant_freqs[i+1]):
                
                # Convert index to frequency
                actual_index = start_idx + i
                frequency = actual_index * freq_resolution
                magnitude = relevant_freqs[i] / len(fft_data)
                
                peaks.append((frequency, magnitude))
        
        # Sort peaks by magnitude (strongest first)
        peaks.sort(key=lambda x: x[1], reverse=True)
        
        # Limit to top 5 peaks to avoid noise
        peaks = peaks[:5]
        
        return peaks, np.max(relevant_freqs) / len(fft_data)
    
    def analyze_frequency_ranges(self, peaks):
        """Analyze detected peaks to identify frequency ranges."""
        if len(peaks) < 2:
            return []
        
        frequencies = [freq for freq, mag in peaks]
        ranges = []
        
        # Sort frequencies
        frequencies.sort()
        
        # Find gaps and group frequencies into ranges
        current_range_start = frequencies[0]
        current_range_end = frequencies[0]
        
        for i in range(1, len(frequencies)):
            freq = frequencies[i]
            
            # If frequency is close to previous (within 100 Hz), extend range
            if freq - current_range_end <= 100:
                current_range_end = freq
            else:
                # Gap found, save current range
                if current_range_end > current_range_start:
                    ranges.append((current_range_start, current_range_end))
                current_range_start = freq
                current_range_end = freq
        
        # Add final range
        if current_range_end > current_range_start:
            ranges.append((current_range_start, current_range_end))
        
        return ranges
    
    def identify_pattern(self, peaks, ranges):
        """Identify which sound.py pattern is being detected."""
        if not peaks:
            return []
        
        detected_patterns = []
        frequencies = [freq for freq, mag in peaks]
        
        # Check single frequencies
        for freq in frequencies:
            for sig_freq, label in self.frequency_signatures.items():
                if isinstance(sig_freq, (int, float)):
                    if abs(freq - sig_freq) < 20:  # Within 20 Hz tolerance
                        detected_patterns.append(label)
                        break
        
        # Check frequency ranges
        for range_start, range_end in ranges:
            for sig_range, label in self.frequency_signatures.items():
                if isinstance(sig_range, tuple):
                    sig_start, sig_end = sig_range
                    # Check if detected range overlaps with signature range
                    if (range_start <= sig_end and range_end >= sig_start):
                        detected_patterns.append(label)
                        break
        
        # Check for DTMF patterns (specific frequency combinations)
        if len(frequencies) >= 2:
            # Sort frequencies for consistent comparison
            sorted_freqs = sorted(frequencies)
            
            # Check for DTMF patterns
            dtmf_patterns = {
                (697, 1336): "DTMF 2",
                (770, 1336): "DTMF 5", 
                (852, 1336): "DTMF 8",
                (941, 1336): "DTMF 0",
                (697, 1477): "DTMF 3",
                (770, 1477): "DTMF 6",
                (770, 1209): "DTMF 4",
            }
            
            for dtmf_freqs, dtmf_label in dtmf_patterns.items():
                if len(sorted_freqs) >= 2:
                    # Check if first two frequencies match DTMF pattern
                    if (abs(sorted_freqs[0] - dtmf_freqs[0]) < 30 and 
                        abs(sorted_freqs[1] - dtmf_freqs[1]) < 30):
                        detected_patterns.append(dtmf_label)
        
        return list(set(detected_patterns))  # Remove duplicates
    
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
        print("Mode: Single Freq | Multiple Freqs | Frequency Ranges")
        print("-" * 70)
        
        last_peaks = []
        stable_count = 0
        
        try:
            while self.is_running:
                # Read audio data
                audio_data = np.frombuffer(
                    self.stream.read(self.chunk_size, exception_on_overflow=False),
                    dtype=np.float32
                )
                
                # Detect multiple frequencies and ranges
                peaks, max_magnitude = self.detect_frequency_range(audio_data)
                
                # More sensitive threshold for sound.py detection
                if max_magnitude > self.magnitude_threshold and len(peaks) > 0:
                    # Filter out very low frequencies
                    peaks = [(freq, mag) for freq, mag in peaks if freq > 50]
                    
                    if len(peaks) > 0:
                        # Analyze frequency ranges
                        ranges = self.analyze_frequency_ranges(peaks)
                        
                        # Identify patterns
                        detected_labels = self.identify_pattern(peaks, ranges)
                        
                        # Check if detection is stable
                        if len(peaks) == len(last_peaks):
                            freq_diffs = [abs(p1[0] - p2[0]) for p1, p2 in zip(peaks, last_peaks)]
                            if all(diff < 20 for diff in freq_diffs):  # Within 20 Hz
                                stable_count += 1
                                status = f"STABLE({stable_count})" if stable_count > 1 else "DETECTED"
                            else:
                                stable_count = 0
                                status = "DETECTED"
                        else:
                            stable_count = 0
                            status = "DETECTED"
                        
                        last_peaks = peaks.copy()
                        
                        # Display results
                        if len(peaks) == 1:
                            # Single frequency
                            freq, mag = peaks[0]
                            note = self.frequency_to_note(freq)
                            print(f"Single: {freq:6.1f} Hz | {mag:6.4f} | {note:>4} | {status}")
                        
                        elif len(peaks) > 1 and len(ranges) == 1:
                            # Frequency range detected
                            range_start, range_end = ranges[0]
                            avg_mag = np.mean([mag for freq, mag in peaks])
                            print(f"Range:  {range_start:6.1f}-{range_end:6.1f} Hz | {avg_mag:6.4f} | {len(peaks)} peaks | {status}")
                        
                        elif len(peaks) > 1:
                            # Multiple frequencies
                            freq_str = ", ".join([f"{freq:.0f}" for freq, mag in peaks[:3]])  # Show first 3
                            if len(peaks) > 3:
                                freq_str += f" (+{len(peaks)-3} more)"
                            avg_mag = np.mean([mag for freq, mag in peaks])
                            print(f"Multi:  {freq_str:>20} Hz | {avg_mag:6.4f} | {len(peaks)} freqs | {status}")
                        
                        # Show ranges if detected
                        if ranges:
                            for i, (start_freq, end_freq) in enumerate(ranges):
                                if i == 0:
                                    print(f"Ranges: {start_freq:6.1f}-{end_freq:6.1f} Hz")
                                else:
                                    print(f"        {start_freq:6.1f}-{end_freq:6.1f} Hz")
                        
                        # Show detected patterns
                        if detected_labels:
                            print(f"Patterns: {', '.join(detected_labels)}")
                
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