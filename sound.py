import pygame
import numpy as np
import time

class SoundPlayer:
    def __init__(self, sample_rate=44100):
        """Initialize the sound player with a given sample rate."""
        pygame.mixer.init(frequency=sample_rate, size=-16, channels=1, buffer=512)
        self.sample_rate = sample_rate
        
    def generate_tone(self, frequency, duration, volume=0.5):
        """Generate a sine wave tone at the specified frequency and duration."""
        # Calculate the number of samples needed
        num_samples = int(self.sample_rate * duration)
        
        # Generate time array
        t = np.linspace(0, duration, num_samples, False)
        
        # Generate sine wave
        tone = np.sin(2 * np.pi * frequency * t)
        
        # Apply volume
        tone = tone * volume
        
        # Convert to 16-bit integer format
        tone = (tone * 32767).astype(np.int16)
        
        return tone
    
    def play_tone(self, frequency, duration=1.0, volume=0.5):
        """Play a tone at the specified frequency for the given duration."""
        # Generate the tone
        tone = self.generate_tone(frequency, duration, volume)
        
        # Create a Sound object from the numpy array
        sound = pygame.sndarray.make_sound(tone)
        
        # Play the sound
        sound.play()
        
        # Wait for the sound to finish
        time.sleep(duration)
        
    def play_scale(self, base_frequency=440, scale_type='major'):
        """Play a musical scale starting from the base frequency."""
        if scale_type == 'major':
            # Major scale intervals: 1, 1, 0.5, 1, 1, 1, 0.5
            intervals = [1, 9/8, 5/4, 4/3, 3/2, 5/3, 15/8, 2]
        elif scale_type == 'minor':
            # Natural minor scale intervals
            intervals = [1, 9/8, 6/5, 4/3, 3/2, 8/5, 9/5, 2]
        else:
            intervals = [1, 9/8, 5/4, 4/3, 3/2, 5/3, 15/8, 2]
        
        for i, interval in enumerate(intervals):
            frequency = base_frequency * interval
            print(f"Playing note {i+1}: {frequency:.1f} Hz")
            self.play_tone(frequency, duration=0.5, volume=0.3)
            time.sleep(0.1)  # Small pause between notes
    
    def play_chord(self, frequencies, duration=2.0, volume=0.3):
        """Play multiple frequencies simultaneously as a chord."""
        # Generate individual tones
        tones = []
        for freq in frequencies:
            tone = self.generate_tone(freq, duration, volume/len(frequencies))
            tones.append(tone)
        
        # Mix the tones together
        mixed_tone = np.sum(tones, axis=0)
        
        # Normalize to prevent clipping
        max_val = np.max(np.abs(mixed_tone))
        if max_val > 32767:
            mixed_tone = (mixed_tone / max_val * 32767).astype(np.int16)
        else:
            mixed_tone = mixed_tone.astype(np.int16)
        
        # Play the mixed tone
        sound = pygame.sndarray.make_sound(mixed_tone)
        sound.play()
        time.sleep(duration)
    
    def play_frequency(self, frequency, duration, volume=0.5):
        """Play a tone at the specified frequency for 0.05 seconds."""
        # Check if frequency is a list/array or single value
        if isinstance(frequency, (list, tuple, np.ndarray)):
            # Multiple frequencies - play as chord
            self.play_chord(frequency, duration, volume=volume)
        else:
            # Single frequency - play as tone
            self.play_tone(frequency, duration, volume=volume)
    
    def play_freq_rand(self, ranges, num_numbers=5, duration=0.05, volume=0.5, step=1):
        """Play random frequencies within specified range(s) simultaneously.
        
        Args:
            ranges: Array of ranges [[low1, high1], [low2, high2], ...] OR single range [low, high]
            num_numbers: Number of random frequencies to generate per range (default 5)
            duration: Duration to play the frequencies (default 0.05 seconds)
            volume: Volume level (default 0.5)
            step: Number of times to re-randomize within the duration (default 1)
        """
        import random
        
        # Ensure ranges is always a list of lists
        if isinstance(ranges[0], (int, float)):
            # Single range provided as [low, high]
            ranges = [ranges]
        
        if step == 1:
            # Generate random frequencies within all ranges
            random_frequencies = []
            for range_low, range_high in ranges:
                for _ in range(num_numbers):
                    freq = random.uniform(range_low, range_high)
                    random_frequencies.append(freq)
            
            # Play all random frequencies simultaneously
            self.play_frequency(random_frequencies, duration, volume)
        else:
            # Calculate step duration
            step_duration = duration / step
            
            # Play multiple steps of random frequencies
            for i in range(step):
                # Generate new random frequencies for this step
                random_frequencies = []
                for range_low, range_high in ranges:
                    for _ in range(num_numbers):
                        freq = random.uniform(range_low, range_high)
                        random_frequencies.append(freq)
                
                # Play this step's frequencies
                self.play_frequency(random_frequencies, step_duration, volume)

def main():
    """Main function to demonstrate the sound player."""
    player = SoundPlayer()

    # Dial tone
    player.play_frequency([350, 440], 1.5)

    # Dial 2053648254
    player.play_frequency([697, 1336, 350, 440], 0.2) # 2
    player.play_frequency([350, 440], 0.07)
    player.play_frequency([941, 1336, 350, 440], 0.2) # 0
    player.play_frequency([350, 440], 0.07)
    player.play_frequency([770, 1336], 0.2) # 5
    time.sleep(0.07)
    player.play_frequency([697, 1477], 0.2) # 3
    time.sleep(0.07)
    player.play_frequency([770, 1477], 0.2) # 6
    time.sleep(0.07)
    player.play_frequency([770, 1209], 0.2) # 4
    time.sleep(0.07)
    player.play_frequency([852, 1336], 0.2) # 8
    time.sleep(0.07)
    player.play_frequency([697, 1336], 0.2) # 2
    time.sleep(0.07)
    player.play_frequency([770, 1336], 0.2) # 5
    time.sleep(0.07)
    player.play_frequency([770, 1209], 0.2) # 4

    # Pause
    time.sleep(2)

    # init
    player.play_frequency([1400, 2000], 0.3, volume=0.1)

    # CRe
    player.play_frequency(430, 0.015)
    time.sleep(0.08)

    # resp
    player.play_frequency([1600, 2250], 0.3)
    player.play_frequency([1600, 2250, 1900], 0.055)
    player.play_frequency([1600, 2250, 1750], 0.025)

    # CL
    player.play_frequency(975, 0.1)
    player.play_freq_rand([[950, 1300]], num_numbers=3, duration=0.9, volume=0.5, step=45)
    player.play_frequency(975, 0.05)

    # Pause
    time.sleep(0.07)

    # MS
    player.play_frequency(1650, 0.08)
    player.play_freq_rand([[1600, 2000]], num_numbers=3, duration=0.85, volume=0.5, step=50)
    player.play_frequency(1650, 0.02)

    # ACK
    player.play_frequency(975, 0.09)
    player.play_freq_rand([[950, 1300]], num_numbers=3, duration=0.15, volume=0.5, step=17)
    player.play_frequency(975, 0.03)

    # Pause
    time.sleep(1)

    # ANSam
    player.play_freq_rand([[2145, 2155]], num_numbers=3, duration=1, volume=0.5, step=3)
    player.play_freq_rand([[850, 1450], [2145, 2155]], num_numbers=3, duration=1, volume=0.5, step=55)
    player.play_freq_rand([[850, 1450], [1600, 2000]], num_numbers=3, duration=1, volume=0.5, step=65)

    # INFO
    player.play_freq_rand([[750, 1750], [1950, 2750]], num_numbers=3, duration=0.15, volume=0.5, step=20)
    player.play_freq_rand([[2745, 2755], [1950, 2750]], num_numbers=3, duration=0.07, volume=0.5, step=12)
    player.play_freq_rand([[1250, 1255], [2750, 2755]], num_numbers=3, duration=0.07, volume=0.5, step=12)

    # L1L2
    player.play_freq_rand([[100, 100], [300, 300], [500, 500], [700, 700], [900, 900], [1010, 1010], [1450, 1450], [1650, 1650], [1850, 1850], [1925, 1925], [2125, 2125], [2325, 2325], [2600, 2600], [2800, 2800], [3000, 3000], [3200, 3200], [3400, 3400], [3600, 3600], [3800, 3800]], num_numbers=1, duration=0.45, volume=0.8, step=1)
    player.play_freq_rand([[1250, 1255], [2750, 2755]], num_numbers=3, duration=0.1, volume=0.8, step=12)
    player.play_freq_rand([[100, 100], [300, 300], [500, 500], [700, 700], [900, 900], [1010, 1010], [1450, 1450], [1650, 1650], [1850, 1850], [1925, 1925], [2125, 2125], [2325, 2325], [2600, 2600], [2800, 2800], [3000, 3000], [3200, 3200], [3400, 3400], [3600, 3600], [3800, 3800]], num_numbers=1, duration=0.45, volume=0.8, step=1)

    # Data
    player.play_freq_rand([[100, 3000]], num_numbers=3, duration=4, volume=0.35, step=1500)
    player.play_freq_rand([[100, 3000]], num_numbers=3, duration=4, volume=0.35, step=300)

if __name__ == "__main__":
    main()
