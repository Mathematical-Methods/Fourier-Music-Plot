import librosa
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading
import time
import pygame

def play_audio(audio_file):
    """Play the audio file using pygame."""
    pygame.mixer.init()           # Initialize the mixer module
    pygame.mixer.music.load(audio_file)  # Load the MP3 file
    pygame.mixer.music.play()     # Start playback
    while pygame.mixer.music.get_busy():  # Wait for playback to finish
        pygame.time.Clock().tick(10)  # Control the loop speed

def plot_frequency_paths(audio_file, N=10, window_size=1024, hop_size=256, duration=None):
    """
    Plots the paths of the top N frequency components of a song in the complex plane,
    synchronized with audio playback.
    
    Parameters:
    - audio_file: Path to the audio file (e.g., .wav or .mp3)
    - N: Number of frequency paths to plot (default: 5)
    - window_size: STFT window size in samples (default: 1024)
    - hop_size: STFT hop size in samples (default: 256)
    - duration: Duration of audio to process in seconds (default: None, full file)
    """
    # Load audio
    y, sr = librosa.load(audio_file, duration=duration)
    
    # Calculate timing parameters
    time_per_frame = hop_size / sr  # Time per STFT frame
    window_duration = 0.3  # Duration of the sliding window in seconds
    window_frames = int(window_duration / time_per_frame)  # Frames in the window
    
    print(f"Loaded audio with sample rate {sr} Hz, duration {len(y)/sr:.2f} seconds")
    
    # Compute STFT
    stft = librosa.stft(y, n_fft=window_size, hop_length=hop_size)
    magnitudes = np.abs(stft)
    print(f"STFT computed: {stft.shape[0]} frequency bins, {stft.shape[1]} time frames")
    
    # Compute average magnitude per frequency bin
    avg_magnitudes = np.mean(magnitudes, axis=1)
    
    # Select top N frequency bins
    top_k_indices = np.argsort(avg_magnitudes)[::-1][:N]
    frequencies = librosa.fft_frequencies(sr=sr, n_fft=window_size)
    top_frequencies = frequencies[top_k_indices]
    print(f"Top {N} frequencies: {top_frequencies[:N]} Hz")
    
    # Extract STFT coefficients for selected frequencies
    selected_stft = stft[top_k_indices, :]
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title(f"Paths of Top {N} Frequency Components")
    ax.set_xlabel("Real Part")
    ax.set_ylabel("Imaginary Part")
    ax.set_aspect('equal')
    ax.grid(True)
    
    # Initialize lines and points for each frequency path
    lines = [ax.plot([], [], lw=1, label=f"{freq:.1f} Hz")[0] for freq in top_frequencies]
    points = [ax.plot([], [], 'o', ms=4)[0] for _ in range(N)]
    
    # Set plot limits based on maximum magnitude
    max_mag = np.max(magnitudes[top_k_indices]) * 1.1  # Add 10% padding
    ax.set_xlim(-max_mag, max_mag)
    ax.set_ylim(-max_mag, max_mag)
    ax.legend()
    
    # Start audio playback in a separate thread
    audio_thread = threading.Thread(target=play_audio, args=(audio_file,))
    audio_thread.start()
    
    # Record the start time of the animation
    start_time = time.time()
    
    def animate(frame):
        """Update the animation based on elapsed time."""
        current_time = time.time() - start_time
        frame = int(current_time / time_per_frame)  # Current frame based on time
        if frame >= selected_stft.shape[1]:
            return lines + points  # Stop updating if beyond audio length
        
        # Compute the start index for the 1-second window
        start = max(0, frame - window_frames + 1)
        for i, line in enumerate(lines):
            # Plot the last 1 second of the path
            path = selected_stft[i, start:frame+1]
            line.set_data(np.real(path), np.imag(path))
            # Update the current position point
            points[i].set_data([np.real(selected_stft[i, frame])], 
                              [np.imag(selected_stft[i, frame])])
        return lines + points
    
    def init():
        """Initialize the plot with empty data."""
        for line, point in zip(lines, points):
            line.set_data([], [])
            point.set_data([], [])
        return lines + points
    
    # Create animation with a minimal interval for real-time updates
    anim = FuncAnimation(fig, animate, init_func=init, frames=None, 
                         interval=1, blit=True)
    
    plt.show()

# Example usage
if __name__ == "__main__":
    # Replace with your audio file path
    audio_file = "/home/unknown/Downloads/What (ever) (feat. SOJUDA)-yt.savetube.me.mp3"
    plot_frequency_paths(audio_file, N=10, duration=None)
