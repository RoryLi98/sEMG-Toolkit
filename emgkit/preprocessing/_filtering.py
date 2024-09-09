"""
Functions implementing filters.


Copyright 2023 Mattia Orlandi

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from .._base import Signal, signal_to_array


def lowpass_filter(x: Signal, cut: float, fs: float, order: int = 2) -> np.ndarray:
    """
    Apply a Butterworth lowpass filter on the given signal.

    Parameters
    ----------
    x : Signal
        A signal with shape (n_samples, n_channels).
    cut : float
        Higher bound for frequency band.
    fs : float
        Sampling frequency.
    order : int, default=2
        Order of the Butterworth filter.

    Returns
    -------
    ndarray
        Filtered signal with shape (n_samples, n_channels).
    """
    # Convert input to array
    x_array = signal_to_array(x)
    # Create and apply filter
    sos = signal.butter(order, cut, btype="lowpass", output="sos", fs=fs)
    return signal.sosfiltfilt(sos, x_array, axis=0).astype(x_array.dtype)


def highpass_filter(x: Signal, cut: float, fs: float, order: int = 2) -> np.ndarray:
    """
    Apply a Butterworth highpass filter on the given signal.

    Parameters
    ----------
    x : Signal
        A signal with shape (n_samples, n_channels).
    cut : float
        Lower bound for frequency band.
    fs : float
        Sampling frequency.
    order : int, default=2
        Order of the Butterworth filter.

    Returns
    -------
    ndarray
        Filtered signal with shape (n_samples, n_channels).
    """
    # Convert input to array
    x_array = signal_to_array(x)
    # Create and apply filter
    sos = signal.butter(order, cut, btype="highpass", output="sos", fs=fs)
    return signal.sosfiltfilt(sos, x_array, axis=0).astype(x_array.dtype)


def bandpass_filter(
    x: Signal,
    low_cut: float,
    high_cut: float,
    fs: float,
    order: int = 2,
) -> np.ndarray:
    """
    Apply a Butterworth bandpass filter on the given signal.

    Parameters
    ----------
    x : Signal
        A signal with shape (n_samples, n_channels).
    low_cut : float
        Lower bound for frequency band.
    high_cut : float
        Higher bound for frequency band.
    fs : float
        Sampling frequency.
    order : int, default=2
        Order of the Butterworth filter.

    Returns
    -------
    ndarray
        Filtered signal with shape (n_samples, n_channels).
    """
    # Convert input to array
    x_array = signal_to_array(x)
    # Create and apply filter
    sos = signal.butter(
        order, (low_cut, high_cut), btype="bandpass", output="sos", fs=fs
    )
    # return signal.sosfiltfilt(sos, x_array, axis=0).astype(x_array.dtype)

    padlen = 3 * (order * sos.shape[0] - 1)
    if x_array.shape[0] > padlen:
        return signal.sosfiltfilt(sos, x_array, axis=0).astype(x_array.dtype)
    else:
        return signal.sosfilt(sos, x_array, axis=0).astype(x_array.dtype)


def bandstop_filter(
    x: Signal,
    low_cut: float,
    high_cut: float,
    fs: float,
    order: int = 2,
) -> np.ndarray:
    """
    Apply a Butterworth bandstop filter on the given signal.

    Parameters
    ----------
    x : Signal
        A signal with shape (n_samples, n_channels).
    low_cut : float
        Lower bound for frequency band.
    high_cut : float
        Higher bound for frequency band.
    fs : float
        Sampling frequency.
    order : int, default=2
        Order of the Butterworth filter.

    Returns
    -------
    ndarray
        Filtered signal with shape (n_samples, n_channels).
    """
    # Convert input to array
    x_array = signal_to_array(x)
    # Create and apply filter
    sos = signal.butter(
        order, (low_cut, high_cut), btype="bandstop", output="sos", fs=fs
    )
    return signal.sosfiltfilt(sos, x_array, axis=0).astype(x_array.dtype)


def notch_filter(
    x: np.ndarray,
    exclude_freqs: Sequence[float],
    fs: float,
    exclude_harmonics: bool = False,
    max_harmonic: float | None = None,
    q: float = 30.0,
) -> np.ndarray:
    """Apply a notch filter on the given signal.

    Parameters
    ----------
    x : ndarray
        Signal with shape (n_channels, n_samples).
    exclude_freqs : sequence of floats
        Frequencies to exclude.
    fs : float
        Sampling frequency.
    exclude_harmonics : bool, default=False
        Whether to exclude all the harmonics, too.
    max_harmonic : float or None, default=None
        Maximum harmonic to exclude.
    q : float, default=30.0
        Quality factor of the filters.

    Returns
    -------
    ndarray
        Filtered signal with shape (n_channels, n_samples).
    """

    def find_multiples(base: float, limit: float) -> list[float]:
        last_mult = int(round(limit / base))
        return [base * i for i in range(1, last_mult + 1)]

    # Find harmonics, if required
    if exclude_harmonics:
        if max_harmonic is None:
            max_harmonic = fs // 2
        exclude_freqs_set = set(
            [f2 for f1 in exclude_freqs for f2 in find_multiples(f1, max_harmonic)]
        )
    else:
        exclude_freqs_set = set(exclude_freqs)

    # Apply series of notch filters
    for freq in exclude_freqs_set:
        b, a = signal.iirnotch(freq, q, fs)
        x = signal.filtfilt(b, a, x)

    return x.copy()


def plot_all_channels_frequency_spectrum(
    x: np.ndarray, fs: float, figsize: tuple = (12, 2)
) -> None:
    """
    Plot the frequency spectrum of all channels in a given multi-channel signal.

    Parameters
    ----------
    x : ndarray
        Multi-channel signal with shape (n_channels, n_samples).
    fs : float
        Sampling frequency.
    figsize : tuple, optional
        Size of the figure (width, height) in inches. Default is (12, 2).

    Returns
    -------
    None
        This function doesn't return anything; it plots the frequency spectrums of all channels.
    """
    # Ensure that the signal is two-dimensional
    if x.ndim != 2:
        raise ValueError(
            "Input signal must be a 2D array with shape (n_channels, n_samples)."
        )

    n_channels, n_samples = x.shape

    # Compute the Fourier Transform for each channel
    freqs = np.fft.rfftfreq(n_samples, d=1 / fs)

    # Determine the number of rows needed for the grid, with 2 subplots per row
    n_cols = 2
    n_rows = int(np.ceil(n_channels / n_cols))

    # Plot the spectrum for each channel
    plt.figure(figsize=(figsize[0], figsize[1] * n_rows))
    for i in range(n_channels):
        freq_spectrum = np.fft.rfft(x[i])

        plt.subplot(n_rows, n_cols, i + 1)  # Adjust the index for the subplot
        plt.plot(freqs, np.abs(freq_spectrum))
        plt.title(f"Frequency Spectrum - Channel {i+1}")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude")
        plt.grid(True)

    plt.tight_layout()
    plt.show()
