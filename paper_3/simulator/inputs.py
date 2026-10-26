"""Official IWA 15-column BSM1 input to explicitly named Python layout."""

from pathlib import Path

import numpy as np

ASM1_NAMES = ("SI", "SS", "XI", "XS", "XBH", "XBA", "XP", "SO", "SNO", "SNH", "SND", "XND", "SALK")

def convert_official(data: np.ndarray) -> np.ndarray:
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] != 15 or len(data) < 2:
        raise ValueError("Expected at least two rows of official 15-column BSM1 data")
    if not np.isfinite(data).all() or np.any(data[:, 1:] < 0):
        raise ValueError("Concentrations and flow must be finite and nonnegative")
    if data[0, 0] != 0 or np.any(np.diff(data[:, 0]) <= 0):
        raise ValueError("Time must start at zero and increase strictly")
    result = np.zeros((len(data), 22), dtype=np.float64)
    result[:, :14] = data[:, :14]
    result[:, 14] = 0.75 * data[:, [3, 4, 5, 6, 7]].sum(axis=1)
    result[:, 15] = data[:, 14]
    result[:, 16] = 15.0
    return result

def load_official(path: str | Path) -> np.ndarray:
    path = Path(path)
    header = path.read_text().splitlines()[0].lower().split()
    expected = ["t", "si", "ss", "xi", "xs", "xbh", "xba", "xp", "so", "sno", "snh", "snd", "xnd", "salk", "q"]
    if header != expected:
        raise ValueError("Official column header does not match BSM1 schema")
    return convert_official(np.loadtxt(path, skiprows=1))

def constant_input(end: float = 100.01) -> np.ndarray:
    # Technical report Table 5. XP is zero, not omitted or mistaken for oxygen.
    row = [0, 30, 69.50, 51.20, 202.32, 28.17, 0, 0, 0, 0, 31.56, 6.95, 10.59, 7, 18446]
    data = np.array([row, row], dtype=float)
    data[-1, 0] = end
    return convert_official(data)
