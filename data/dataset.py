"""
PyTorch Dataset classes for LIBS Foundation Model.

Includes:
- LIBSDataset: Base dataset class
- MaskedLIBSDataset: Dataset with advanced masking for self-supervised pre-training
- LabeledLIBSDataset: Dataset with labels for supervised fine-tuning

Masking Strategies:
- Random: Individual bins scattered randomly
- Contiguous: Fixed-size blocks placed randomly
- Variable: Mixed block sizes (25, 50, 100, 150 bins)
- Peak-biased: Ensures specified fraction of masks overlap with peaks
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path


class LIBSDataset(Dataset):
    """
    Base dataset class for LIBS spectra.
    
    Args:
        spectra: Array of shape (n_samples, n_bins) or path to .npy file
        transform: Optional transform to apply to spectra
    """
    
    def __init__(
        self,
        spectra: Union[np.ndarray, str, Path],
        transform: Optional[callable] = None,
    ):
        if isinstance(spectra, (str, Path)):
            self.spectra = np.load(spectra)
        else:
            self.spectra = np.asarray(spectra)
        
        self.spectra = self.spectra.astype(np.float32)
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.spectra)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        spectrum = self.spectra[idx]
        
        if self.transform is not None:
            spectrum = self.transform(spectrum)
        
        return torch.from_numpy(spectrum)


class SpectrumDataset(LIBSDataset):
    """Unlabelled spectra as ``{'spectrum': [n_bins]}`` batches (spectral_patch
    pretraining: the window masks are drawn on the GPU by LIBSPretrainModule)."""

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {'spectrum': super().__getitem__(idx)}


class MaskedLIBSDataset(Dataset):
    """
    Dataset with advanced masking for self-supervised pre-training.
    
    Masking Strategies:
    ==================
    1. Random: Individual bins scattered randomly (BERT-style)
    2. Contiguous: Fixed-size blocks (e.g., 50 bins each)
    3. Variable: Mixed block sizes sampled from block_sizes list
    4. Peak-biased: Ensures peak_bias_ratio of masked bins overlap with peaks
    
    BERT-style Token Replacement:
    - 80% of masked positions: replace with 0 (mask token)
    - 10% of masked positions: replace with random value
    - 10% of masked positions: keep unchanged
    
    Args:
        spectra: Array of shape (n_samples, n_bins) or path to .npy file
        mask_ratio: Fraction of positions to mask (default: 0.15)
        mask_token_prob: Probability of using mask token (default: 0.8)
        random_token_prob: Probability of using random value (default: 0.1)
        contiguous_masking: If True, use contiguous block masking
        block_sizes: List of possible block sizes for variable masking (default: [50])
                     If multiple sizes provided, samples randomly from list
        peak_bias_enabled: If True, bias masking toward peak regions
        peak_bias_ratio: Fraction of masked bins that must overlap peaks (default: 0.5)
        peak_threshold: Intensity threshold for peak detection (default: 0.2)
        transform: Optional transform to apply before masking
        seed: Random seed for reproducibility (optional)
    """
    
    def __init__(
        self,
        spectra: Union[np.ndarray, str, Path],
        mask_ratio: float = 0.15,
        mask_token_prob: float = 0.8,
        random_token_prob: float = 0.1,
        contiguous_masking: bool = False,
        block_sizes: Optional[List[int]] = None,
        peak_bias_enabled: bool = False,
        peak_bias_ratio: float = 0.5,
        peak_threshold: float = 0.2,
        transform: Optional[callable] = None,
        seed: Optional[int] = None,
        # Legacy parameter for backward compatibility
        contiguous_mask_size: int = 50,
    ):
        if isinstance(spectra, (str, Path)):
            self.spectra = np.load(spectra)
        else:
            self.spectra = np.asarray(spectra)
        
        self.spectra = self.spectra.astype(np.float32)
        self.n_bins = self.spectra.shape[1]
        
        self.mask_ratio = mask_ratio
        self.mask_token_prob = mask_token_prob
        self.random_token_prob = random_token_prob
        self.contiguous_masking = contiguous_masking
        
        # Block sizes: use provided list or fall back to legacy single size
        if block_sizes is not None:
            self.block_sizes = block_sizes
        else:
            self.block_sizes = [contiguous_mask_size]
        
        # Peak-biased masking
        self.peak_bias_enabled = peak_bias_enabled
        self.peak_bias_ratio = peak_bias_ratio
        self.peak_threshold = peak_threshold
        
        self.transform = transform
        self.rng = np.random.default_rng(seed)

    def _reseed(self, seed: int) -> None:
        """Re-seed the per-instance rng. Called by worker_init_fn so each
        DataLoader worker derives a distinct, reproducible mask stream from
        the master torch seed (which advances per epoch via the sampler)."""
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.spectra)
    
    def _detect_peaks(self, spectrum: np.ndarray) -> np.ndarray:
        """
        Detect peak regions in spectrum.
        
        Returns:
            Boolean array where True indicates peak regions
        """
        # Simple threshold-based peak detection
        # A bin is considered a "peak region" if intensity > threshold
        peak_mask = spectrum > self.peak_threshold
        
        # Expand peak regions slightly to include shoulders
        # This ensures we capture the full peak structure
        expanded_mask = np.zeros_like(peak_mask)
        for i in range(len(peak_mask)):
            if peak_mask[i]:
                # Include ±5 bins around each peak point
                start = max(0, i - 5)
                end = min(len(peak_mask), i + 6)
                expanded_mask[start:end] = True
        
        return expanded_mask
    
    def _create_random_mask(self) -> np.ndarray:
        """Create a random mask selecting mask_ratio fraction of positions."""
        n_masked = int(self.n_bins * self.mask_ratio)
        mask_positions = self.rng.choice(self.n_bins, size=n_masked, replace=False)
        mask = np.zeros(self.n_bins, dtype=bool)
        mask[mask_positions] = True
        return mask
    
    def _create_contiguous_mask(self, spectrum: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Create a mask with contiguous regions of variable sizes.
        
        If peak_bias_enabled, ensures peak_bias_ratio of masked bins are on peaks.
        """
        mask = np.zeros(self.n_bins, dtype=bool)
        n_masked_total = int(self.n_bins * self.mask_ratio)
        
        # Detect peaks if peak-biased masking is enabled
        if self.peak_bias_enabled and spectrum is not None:
            peak_regions = self._detect_peaks(spectrum)
            peak_indices = np.where(peak_regions)[0]
            non_peak_indices = np.where(~peak_regions)[0]
            
            # Calculate how many bins should be on peaks vs off peaks
            n_peak_masked = int(n_masked_total * self.peak_bias_ratio)
            n_non_peak_masked = n_masked_total - n_peak_masked
            
            masked_count = 0
            
            # First, place blocks on peak regions
            if len(peak_indices) > 0 and n_peak_masked > 0:
                masked_count = self._place_blocks_in_region(
                    mask, peak_indices, n_peak_masked
                )
            
            # Then, place remaining blocks anywhere (preferring non-peak to balance)
            remaining = n_masked_total - mask.sum()
            if remaining > 0:
                # Try to place in non-peak regions first
                if len(non_peak_indices) > 0:
                    self._place_blocks_in_region(
                        mask, non_peak_indices, remaining, allow_overlap=False
                    )
                
                # If still not enough, place anywhere
                remaining = n_masked_total - mask.sum()
                if remaining > 0:
                    all_indices = np.arange(self.n_bins)
                    self._place_blocks_in_region(
                        mask, all_indices, remaining, allow_overlap=False
                    )
        else:
            # Standard contiguous masking without peak bias
            self._place_blocks_anywhere(mask, n_masked_total)
        
        return mask
    
    def _place_blocks_in_region(
        self, 
        mask: np.ndarray, 
        valid_indices: np.ndarray, 
        target_count: int,
        allow_overlap: bool = False
    ) -> int:
        """
        Place contiguous blocks within specified region.
        
        Args:
            mask: Mask array to modify in-place
            valid_indices: Indices where blocks can start
            target_count: Target number of bins to mask
            allow_overlap: Whether to allow overlapping with existing mask
            
        Returns:
            Number of new bins masked
        """
        masked_count = 0
        attempts = 0
        max_attempts = 100
        
        while masked_count < target_count and attempts < max_attempts:
            # Sample a random block size
            block_size = self.rng.choice(self.block_sizes)
            
            # Find valid start positions within the region
            valid_starts = valid_indices[valid_indices <= self.n_bins - block_size]
            
            if len(valid_starts) == 0:
                attempts += 1
                continue
            
            # Pick a random start position
            start = self.rng.choice(valid_starts)
            end = min(start + block_size, self.n_bins)
            
            # Check for overlap if not allowed
            if not allow_overlap and mask[start:end].any():
                attempts += 1
                continue
            
            # Apply mask
            new_masked = (~mask[start:end]).sum()
            mask[start:end] = True
            masked_count += new_masked
            attempts = 0  # Reset attempts on success
        
        return masked_count
    
    def _place_blocks_anywhere(self, mask: np.ndarray, target_count: int):
        """Place contiguous blocks anywhere in the spectrum."""
        masked_count = 0
        attempts = 0
        max_attempts = 100
        
        while masked_count < target_count and attempts < max_attempts:
            # Sample a random block size
            block_size = self.rng.choice(self.block_sizes)
            
            # Random start position
            if self.n_bins <= block_size:
                start = 0
            else:
                start = self.rng.integers(0, self.n_bins - block_size)
            end = min(start + block_size, self.n_bins)
            
            # Check if region overlaps with existing mask
            if not mask[start:end].any():
                mask[start:end] = True
                masked_count += (end - start)
                attempts = 0
            else:
                attempts += 1
    
    def _apply_masking(
        self, 
        spectrum: np.ndarray, 
        mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply BERT-style masking to spectrum.
        
        Returns:
            masked_spectrum: Spectrum with masked positions modified
            mask_type: Array indicating what was done at each position
                      0 = not masked, 1 = mask token, 2 = random, 3 = unchanged
        """
        masked_spectrum = spectrum.copy()
        mask_type = np.zeros(self.n_bins, dtype=np.int64)
        
        masked_indices = np.where(mask)[0]
        
        for idx in masked_indices:
            rand = self.rng.random()
            
            if rand < self.mask_token_prob:
                # Replace with 0 (mask token)
                masked_spectrum[idx] = 0.0
                mask_type[idx] = 1
            elif rand < self.mask_token_prob + self.random_token_prob:
                # Replace with random value from spectrum distribution
                masked_spectrum[idx] = self.rng.uniform(0, 1)
                mask_type[idx] = 2
            else:
                # Keep unchanged
                mask_type[idx] = 3
        
        return masked_spectrum, mask_type
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a masked sample.
        
        Returns:
            Dictionary with:
            - 'input': Masked spectrum [n_bins]
            - 'target': Original spectrum [n_bins]
            - 'mask': Boolean mask indicating masked positions [n_bins]
            - 'mask_type': Type of masking applied at each position [n_bins]
        """
        spectrum = self.spectra[idx].copy()
        
        if self.transform is not None:
            spectrum = self.transform(spectrum)
        
        # Create mask
        if self.contiguous_masking:
            mask = self._create_contiguous_mask(spectrum)
        else:
            mask = self._create_random_mask()
        
        # Apply masking
        masked_spectrum, mask_type = self._apply_masking(spectrum, mask)
        
        return {
            'input': torch.from_numpy(masked_spectrum),
            'target': torch.from_numpy(spectrum),
            'mask': torch.from_numpy(mask),
            'mask_type': torch.from_numpy(mask_type),
        }
    
    def get_masking_stats(self, n_samples: int = 100) -> Dict[str, float]:
        """
        Compute statistics about masking coverage.
        
        Args:
            n_samples: Number of samples to analyze
            
        Returns:
            Dictionary with masking statistics
        """
        total_masked = 0
        total_peak_masked = 0
        total_peaks = 0
        
        indices = self.rng.choice(len(self), size=min(n_samples, len(self)), replace=False)
        
        for idx in indices:
            spectrum = self.spectra[idx]
            
            if self.contiguous_masking:
                mask = self._create_contiguous_mask(spectrum)
            else:
                mask = self._create_random_mask()
            
            peak_regions = self._detect_peaks(spectrum)
            
            total_masked += mask.sum()
            total_peak_masked += (mask & peak_regions).sum()
            total_peaks += peak_regions.sum()
        
        return {
            'avg_masked_bins': total_masked / n_samples,
            'avg_masked_ratio': total_masked / (n_samples * self.n_bins),
            'peak_coverage': total_peak_masked / max(total_masked, 1),
            'avg_peak_bins': total_peaks / n_samples,
        }


class LabeledLIBSDataset(Dataset):
    """
    Dataset with labels for supervised fine-tuning.
    
    Supports both classification (dominant class) and regression (concentrations).
    
    Args:
        spectra: Array of shape (n_samples, n_bins) or path to .npy file
        labels: Array of shape (n_samples,) with class labels, or path to .npy file
        concentrations: Optional array of shape (n_samples, n_classes), or path to .npy file
        transform: Optional transform to apply to spectra
    """
    
    def __init__(
        self,
        spectra: Union[np.ndarray, str, Path],
        labels: Union[np.ndarray, str, Path],
        concentrations: Optional[Union[np.ndarray, str, Path]] = None,
        transform: Optional[callable] = None,
    ):
        # Load spectra
        if isinstance(spectra, (str, Path)):
            self.spectra = np.load(spectra)
        else:
            self.spectra = np.asarray(spectra)
        self.spectra = self.spectra.astype(np.float32)
        
        # Load labels
        if isinstance(labels, (str, Path)):
            self.labels = np.load(labels)
        else:
            self.labels = np.asarray(labels)
        self.labels = self.labels.astype(np.int64)
        
        # Load concentrations (optional)
        if concentrations is not None:
            if isinstance(concentrations, (str, Path)):
                self.concentrations = np.load(concentrations)
            else:
                self.concentrations = np.asarray(concentrations)
            self.concentrations = self.concentrations.astype(np.float32)
        else:
            self.concentrations = None
        
        self.transform = transform
        
        # Validate shapes
        assert len(self.spectra) == len(self.labels), \
            f"Spectra and labels must have same length: {len(self.spectra)} vs {len(self.labels)}"
        if self.concentrations is not None:
            assert len(self.spectra) == len(self.concentrations), \
                f"Spectra and concentrations must have same length"
    
    def __len__(self) -> int:
        return len(self.spectra)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a labeled sample.
        
        Returns:
            Dictionary with:
            - 'spectrum': Input spectrum [n_bins]
            - 'label': Class label (scalar)
            - 'concentrations': Concentration values [n_classes] (if available)
        """
        spectrum = self.spectra[idx]
        
        if self.transform is not None:
            spectrum = self.transform(spectrum)
        
        result = {
            'spectrum': torch.from_numpy(spectrum),
            'label': torch.tensor(self.labels[idx], dtype=torch.long),
        }
        
        if self.concentrations is not None:
            result['concentrations'] = torch.from_numpy(self.concentrations[idx])
        
        return result


class LineFeaturesDataset(Dataset):
    """Read precomputed [n_lines, 6] features from HDF5 by spectrum index."""

    def __init__(self, features_path: str, indices: Optional[np.ndarray] = None):
        import h5py
        self.features_path = str(features_path)
        with h5py.File(self.features_path, "r") as f:
            self._n_spectra = int(f.attrs["n_spectra"])
        self.indices = np.arange(self._n_spectra) if indices is None else np.asarray(indices, dtype=np.int64)
        self._file = None

    def _open(self):
        if self._file is None:
            import h5py
            self._file = h5py.File(self.features_path, "r")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> torch.Tensor:
        self._open()
        spec_idx = int(self.indices[idx])
        return torch.from_numpy(self._file["features"][spec_idx].astype(np.float32))


class MaskedLineTokenDataset(LineFeaturesDataset):
    """
    Self-supervised line-token batches for MIP on Voigt fit channels.

    Masks a fraction of lines with valid fits; zeros max_intensity & FWHM in input.
    """

    def __init__(
        self,
        features_path: str,
        indices: Optional[np.ndarray] = None,
        mask_ratio: float = 0.15,
        seed: int = 42,
    ):
        super().__init__(features_path, indices=indices)
        self.mask_ratio = mask_ratio
        self.rng = np.random.default_rng(seed)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        from data.line_features import FEAT_FWHM, FEAT_MAX_INT, FEAT_VALID

        feats = super().__getitem__(idx).clone()
        valid = feats[:, FEAT_VALID] > 0.5
        n_valid = int(valid.sum().item())
        if n_valid == 0:
            line_mask = torch.zeros(feats.size(0), dtype=torch.bool)
        else:
            n_mask = max(1, int(n_valid * self.mask_ratio))
            valid_idx = torch.where(valid)[0].numpy()
            chosen = self.rng.choice(valid_idx, size=min(n_mask, len(valid_idx)), replace=False)
            line_mask = torch.zeros(feats.size(0), dtype=torch.bool)
            line_mask[chosen] = True

        input_feats = feats.clone()
        input_feats[line_mask, FEAT_MAX_INT] = 0.0
        input_feats[line_mask, FEAT_FWHM] = 0.0

        target = feats[:, :2].clone()  # max_I, FWHM

        return {
            "line_features": input_feats,
            "target": target,
            "mask": line_mask,
            "input": input_feats,  # alias for pretrain module
        }


class LineTokensDataset(Dataset):
    """
    Lazy reader over the pre-baked ``line_tokens_<hash>.h5`` cache built by
    :func:`data.line_tokenization.build_line_tokens_cache`.

    Yields ``{"tokens": [n_lines, n_features], "fit_valid": [n_lines]}`` per
    spectrum index. The HDF5 file is opened lazily (per worker) so this
    dataset is safe to use with ``DataLoader(num_workers > 0)``.
    """

    def __init__(self, tokens_path: str, indices: Optional[np.ndarray] = None):
        import h5py
        self.tokens_path = str(tokens_path)
        with h5py.File(self.tokens_path, "r") as f:
            self._n_spectra = int(f.attrs["n_spectra"])
            self.n_lines = int(f.attrs["n_lines"])
            self.n_features = int(f.attrs["n_features"])
        self.indices = (
            np.arange(self._n_spectra) if indices is None
            else np.asarray(indices, dtype=np.int64)
        )
        self._file = None

    def _open(self):
        if self._file is None:
            import h5py
            self._file = h5py.File(self.tokens_path, "r")

    def __len__(self) -> int:
        return len(self.indices)

    def _read(self, spec_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        self._open()
        toks = self._file["tokens"][spec_idx].astype(np.float32)
        valid = self._file["fit_valid"][spec_idx].astype(np.uint8)
        return toks, valid

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        spec_idx = int(self.indices[idx])
        toks, valid = self._read(spec_idx)
        return {
            "tokens": torch.from_numpy(toks),
            "fit_valid": torch.from_numpy(valid),
        }


class MaskedLineTokensDataset(LineTokensDataset):
    """
    Self-supervised batches for the ``line_token_linear`` model.

    A fraction of *valid* lines is masked. For each masked line:
        - Targets are the original feature values at
          ``target_feature_indices`` (default: max_intensity + FWHM).
        - Input feature values at those same channels are zeroed.
    The encoder receives ``fit_valid`` so attention can still ignore
    failed Voigt fits (independent of masking).
    """

    def __init__(
        self,
        tokens_path: str,
        indices: Optional[np.ndarray] = None,
        mask_ratio: float = 0.15,
        target_feature_indices: Optional[Tuple[int, ...]] = None,
        seed: int = 42,
    ):
        super().__init__(tokens_path, indices=indices)
        from data.line_tokenization import MIP_TARGET_INDICES
        self.mask_ratio = float(mask_ratio)
        self.target_feature_indices = tuple(
            target_feature_indices if target_feature_indices is not None else MIP_TARGET_INDICES
        )
        self.rng = np.random.default_rng(seed)

    def _reseed(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        spec_idx = int(self.indices[idx])
        toks, valid = self._read(spec_idx)
        n_lines = toks.shape[0]
        targets_np = toks[:, list(self.target_feature_indices)].copy()

        valid_idx = np.flatnonzero(valid > 0)
        line_mask = np.zeros(n_lines, dtype=bool)
        if valid_idx.size > 0:
            n_mask = max(1, int(round(valid_idx.size * self.mask_ratio)))
            chosen = self.rng.choice(valid_idx, size=min(n_mask, valid_idx.size), replace=False)
            line_mask[chosen] = True
            for ci in self.target_feature_indices:
                toks[chosen, ci] = 0.0

        return {
            "tokens": torch.from_numpy(toks),
            "fit_valid": torch.from_numpy(valid),
            "target": torch.from_numpy(targets_np),
            "mask": torch.from_numpy(line_mask),
            "input": torch.from_numpy(toks),  # alias for symmetry with intensity path
        }


def _prepare_aux_targets(
    aux_targets: Optional[Dict[str, np.ndarray]],
    n: int,
) -> Dict[str, np.ndarray]:
    """Validate per-spectrum auxiliary targets (already sliced to the split).

    Each value is coerced to a float32 array of length ``n`` (one scalar per
    spectrum, matching ``indices``). Used for the plasma-state labels of the
    ``cf_quantification`` task (``Te``, ``log10_Ne``, ``log10_Nl``,
    ``is_two_zone``, ``has_plasma_labels``); any key is accepted and emitted
    verbatim so the default collate produces one ``[B]`` tensor per key.
    """
    out: Dict[str, np.ndarray] = {}
    if not aux_targets:
        return out
    for key, value in aux_targets.items():
        arr = np.asarray(value, dtype=np.float32)
        if arr.ndim == 0:
            arr = np.full(n, float(arr), dtype=np.float32)
        if arr.shape[0] != n:
            raise ValueError(
                f"aux_targets[{key!r}] has {arr.shape[0]} rows but the dataset has {n} indices"
            )
        out[str(key)] = arr
    return out


class LineTokensLabeledDataset(LineTokensDataset):
    """Labeled fine-tuning with pre-baked tokens + class / concentration targets.

    ``aux_targets`` (optional) is a dict of per-spectrum float arrays already
    sliced to the same split as ``labels`` (e.g. the output of
    ``data.libs_pipeline.extract_plasma_targets`` indexed by the split);
    every key is emitted as a float32 scalar tensor per item.
    """

    def __init__(
        self,
        tokens_path: str,
        labels: np.ndarray,
        concentrations: Optional[np.ndarray] = None,
        indices: Optional[np.ndarray] = None,
        aux_targets: Optional[Dict[str, np.ndarray]] = None,
    ):
        super().__init__(tokens_path, indices=indices)
        self.labels = np.asarray(labels).astype(np.int64)
        self.concentrations = (
            None if concentrations is None
            else np.asarray(concentrations, dtype=np.float32)
        )
        if len(self.labels) != len(self.indices):
            raise ValueError(
                f"labels ({len(self.labels)}) must match indices ({len(self.indices)})"
            )
        self.aux_targets = _prepare_aux_targets(aux_targets, len(self.indices))

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        spec_idx = int(self.indices[idx])
        toks, valid = self._read(spec_idx)
        out = {
            "tokens": torch.from_numpy(toks),
            "fit_valid": torch.from_numpy(valid),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }
        if self.concentrations is not None:
            out["concentrations"] = torch.from_numpy(self.concentrations[idx])
        for key, arr in self.aux_targets.items():
            out[key] = torch.tensor(arr[idx], dtype=torch.float32)
        return out


class LineTokenLabeledDataset(LineFeaturesDataset):
    """Labeled fine-tuning with line features + class/concentration targets.

    ``aux_targets`` behaves as in :class:`LineTokensLabeledDataset`.
    """

    def __init__(
        self,
        features_path: str,
        labels: np.ndarray,
        concentrations: Optional[np.ndarray] = None,
        indices: Optional[np.ndarray] = None,
        aux_targets: Optional[Dict[str, np.ndarray]] = None,
    ):
        super().__init__(features_path, indices=indices)
        self.labels = np.asarray(labels).astype(np.int64)
        self.concentrations = None if concentrations is None else np.asarray(concentrations, dtype=np.float32)
        assert len(self.labels) == len(self.indices)
        self.aux_targets = _prepare_aux_targets(aux_targets, len(self.indices))

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        feats = super().__getitem__(idx)
        out = {
            "line_features": feats,
            "spectrum": feats,  # legacy key for modules that read batch['spectrum']
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }
        if self.concentrations is not None:
            out["concentrations"] = torch.from_numpy(self.concentrations[idx])
        for key, arr in self.aux_targets.items():
            out[key] = torch.tensor(arr[idx], dtype=torch.float32)
        return out


def create_data_loaders(
    train_spectra: np.ndarray,
    val_spectra: np.ndarray,
    batch_size: int = 64,
    mask_ratio: float = 0.15,
    contiguous_masking: bool = False,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create data loaders for pre-training.
    
    Args:
        train_spectra: Training spectra array
        val_spectra: Validation spectra array
        batch_size: Batch size
        mask_ratio: Masking ratio
        contiguous_masking: Whether to use contiguous masking
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory for GPU transfer
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_dataset = MaskedLIBSDataset(
        spectra=train_spectra,
        mask_ratio=mask_ratio,
        contiguous_masking=contiguous_masking,
    )
    
    val_dataset = MaskedLIBSDataset(
        spectra=val_spectra,
        mask_ratio=mask_ratio,
        contiguous_masking=contiguous_masking,
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    return train_loader, val_loader


def create_labeled_data_loaders(
    train_spectra: np.ndarray,
    train_labels: np.ndarray,
    val_spectra: np.ndarray,
    val_labels: np.ndarray,
    train_concentrations: Optional[np.ndarray] = None,
    val_concentrations: Optional[np.ndarray] = None,
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create data loaders for fine-tuning.
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_dataset = LabeledLIBSDataset(
        spectra=train_spectra,
        labels=train_labels,
        concentrations=train_concentrations,
    )
    
    val_dataset = LabeledLIBSDataset(
        spectra=val_spectra,
        labels=val_labels,
        concentrations=val_concentrations,
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    return train_loader, val_loader


def libs_worker_init_fn(worker_id: int) -> None:
    """DataLoader worker initializer.

    PyTorch sets `worker_info.seed` per worker from the main-process torch
    rng (which advances per epoch via the RandomSampler). We use that seed
    to deterministically reseed the dataset's numpy rng so:
      - different workers see different mask streams in the same epoch
      - the same worker sees a different mask stream each epoch
      - the whole sequence is reproducible from the master torch seed
    """
    info = torch.utils.data.get_worker_info()
    if info is None:
        return
    dataset = info.dataset
    # torch worker seed is uint64; numpy default_rng accepts up to 2**32-1 safely.
    seed = info.seed % (2**32)
    if hasattr(dataset, '_reseed'):
        dataset._reseed(seed)


if __name__ == "__main__":
    # Test the datasets
    from synthetic_generator import SyntheticLIBSGenerator
    
    generator = SyntheticLIBSGenerator(seed=42)
    spectra, labels, concentrations = generator.generate_dataset(n_samples=1000)
    
    print("Testing MaskedLIBSDataset...")
    masked_dataset = MaskedLIBSDataset(spectra, mask_ratio=0.15)
    sample = masked_dataset[0]
    print(f"  Input shape: {sample['input'].shape}")
    print(f"  Target shape: {sample['target'].shape}")
    print(f"  Mask shape: {sample['mask'].shape}")
    print(f"  Masked positions: {sample['mask'].sum().item()}")
    
    print("\nTesting LabeledLIBSDataset...")
    labeled_dataset = LabeledLIBSDataset(spectra, labels, concentrations)
    sample = labeled_dataset[0]
    print(f"  Spectrum shape: {sample['spectrum'].shape}")
    print(f"  Label: {sample['label'].item()}")
    print(f"  Concentrations shape: {sample['concentrations'].shape}")
    
    print("\nTesting DataLoaders...")
    train_loader, val_loader = create_data_loaders(
        spectra[:800], spectra[800:], batch_size=32
    )
    batch = next(iter(train_loader))
    print(f"  Batch input shape: {batch['input'].shape}")
    print(f"  Batch target shape: {batch['target'].shape}")
