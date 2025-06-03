import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union
import warnings
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants for TMM normalization
TMM_M_PERCENTILE_LOW = 0.3
TMM_M_PERCENTILE_HIGH = 0.7
TMM_A_PERCENTILE_LOW = 0.05
TMM_A_PERCENTILE_HIGH = 0.95
SCALING_FACTOR = 1_000_000

class CountTableParser:
    """
    A parser for processing e2MPRA data.
    
    This class handles the normalization and log fold change calculation for genomic assay data,
    including lentiMPRA, ATAC-seq, and H3K27ac data.
    """
    
    def __init__(self, 
        sample_label_path: str, 
        mpra_path: str = None, 
        cnt_path: str = None, 
        gDNA_sample: str = 'gDNA', 
        index_col: List[str] = None,
        reps: List[int] = None
    ):
        """
        Initialize the CountTableParser with input file paths and parameters.
        
        Args:
            sample_label_path: Path to sample label file
            mpra_path: Path to MPRA data file
            cnt_path: Path to ATAC and CUT&Tag data file
            gDNA_sample: Name of gDNA sample column
            index_col: List of columns to use as index
            reps: List of replicate numbers
        """
        # Set default values
        if index_col is None:
            index_col = ['feature', 'id', 'replicate']
        if reps is None:
            reps = [1, 2, 3]
            
        self.gDNA_sample = gDNA_sample
        self.index_col = index_col
        self.reps = reps
        self.cnt_assays = ['ATAC', 'H3K27ac']
        
        # Initialize data containers
        self.label: Optional[pd.DataFrame] = None
        self.mpra_ori: Optional[pd.DataFrame] = None
        self.cnt_ori: Optional[pd.DataFrame] = None
        self.tmm_norm: Dict[str, pd.Series] = {}
        self.logFC_reps: Dict[str, pd.Series] = {}
        self.logFC_mean: Dict[str, pd.Series] = {}
        self.processed = False
        
        # Load input data
        self._load_sample_labels(sample_label_path)
        if mpra_path:
            self._load_mpra_data(mpra_path)
        if cnt_path:
            self._load_cnt_data(cnt_path)

        self.process()
    
    def _load_sample_labels(self, sample_label_path: str) -> None:
        """Load sample label data from file."""
        try:
            self.label = pd.read_csv(sample_label_path, sep='\t', index_col='id')
            logger.info(f"Loaded sample labels: {self.label.shape[0]} samples")
        except Exception as e:
            logger.error(f"Failed to load sample labels from {sample_label_path}: {e}")
            raise
    
    def _load_mpra_data(self, mpra_path: str) -> None:
        """Load MPRA data from file."""
        try:
            required_cols = ['name', 'replicate', 'dna_count', 'rna_count', 'n_obs_bc']
            self.mpra_ori = pd.read_csv(
                mpra_path, 
                delimiter='\t', 
                index_col='name',
                usecols=required_cols
            ).drop(index='no_BC', errors='ignore')
            self.mpra_ori.index.name = 'id'
            logger.info(f"Loaded MPRA data: {self.mpra_ori.shape[0]} entries")
        except Exception as e:
            logger.error(f"Failed to load MPRA data from {mpra_path}: {e}")
            raise
    
    def _load_cnt_data(self, cnt_path: str) -> None:
        """Load ATAC and CUT&Tag data from file."""
        try:
            self.cnt_ori = pd.read_csv(cnt_path, delimiter='\t', index_col='id')
            logger.info(f"Loaded ATAC and CUT&Tag data: {self.cnt_ori.shape[0]} entries")
        except Exception as e:
            logger.error(f"Failed to load ATAC and CUT&Tag data from {cnt_path}: {e}")
            raise
    
    def process(self) -> None:
        """
        Main processing function that formats and normalizes all loaded data.
        
        This method orchestrates the entire analysis pipeline including:
        - Data formatting
        - TMM normalization
        - Log fold change calculation
        """
        if self.label is None:
            raise ValueError("Sample labels not loaded")
        
        if self.mpra_ori is not None:
            logger.info("Processing MPRA data...")
            self._process_mpra_data()
        
        if self.cnt_ori is not None:
            logger.info("Processing ATAC and CUT&Tag data...")
            self._process_cnt_data()
        
        # Convert dictionaries to DataFrames
        self.tmm_norm = pd.DataFrame(self.tmm_norm)
        self.logFC_reps = pd.DataFrame(self.logFC_reps)
        self.logFC_mean = pd.DataFrame(self.logFC_mean)
        
        self.processed = True
        logger.info("Processing completed successfully")
    
    def _format_original_data(self, ori_df: pd.DataFrame) -> pd.DataFrame:
        """
        Format original data by joining with sample labels and setting proper index.
        
        Args:
            ori_df: Original data DataFrame
            
        Returns:
            Formatted DataFrame with proper indexing
        """
        if self.label is None:
            raise ValueError("Sample labels not available")
            
        df = ori_df.join(self.label).reset_index()
        df = df.set_index(self.index_col).sort_index()
        return df
    
    def _process_mpra_data(self) -> None:
        """Process MPRA data including read calculation and normalization."""
        if self.mpra_ori is None:
            return
            
        # Format data
        mpra = self._format_original_data(self.mpra_ori)
        
        # Calculate total reads
        mpra = self._calculate_mpra_reads(mpra)
        
        # Normalize data
        normalized = self._normalize_for_replicates(mpra, ['rna_reads'], 'dna_reads')
        
        # Store normalized counts
        self.tmm_norm['rna_count'] = normalized['rna_reads'] / mpra['n_obs_bc']
        self.tmm_norm['dna_count'] = normalized['dna_reads'] / mpra['n_obs_bc']
        
        # Calculate log fold changes
        self._calculate_mpra_logfc()
    
    def _calculate_mpra_reads(self, mpra: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate total DNA and RNA reads from count and barcode data.
        
        Args:
            mpra: MPRA DataFrame with count and barcode information
            
        Returns:
            DataFrame with calculated read counts
        """
        mpra['dna_reads'] = mpra['dna_count'] * mpra['n_obs_bc']
        mpra['rna_reads'] = mpra['rna_count'] * mpra['n_obs_bc']
        return mpra
    
    def _calculate_mpra_logfc(self) -> None:
        """Calculate log fold changes for MPRA data."""
        # Per-replicate log fold change
        self.logFC_reps['lentiMPRA'] = np.log2(
            self.tmm_norm['rna_count'] / self.tmm_norm['dna_count']
        )
        
        # Mean log fold change across replicates
        without_rep = self._get_index_without_replicate()
        rna_sum = self.tmm_norm['rna_count'].groupby(without_rep).sum()
        dna_sum = self.tmm_norm['dna_count'].groupby(without_rep).sum()
        self.logFC_mean['lentiMPRA'] = np.log2(rna_sum / dna_sum)
    
    def _process_cnt_data(self) -> None:
        """Process ATAC and CUT&Tag data including normalization and log fold change calculation."""
        if self.cnt_ori is None:
            return
            
        # Format data and filter positive values
        cnt = self._format_original_data(self.cnt_ori)
        cnt_positive = cnt[cnt > 0]
        
        # Normalize data
        assays_and_gdna = self.cnt_assays + [self.gDNA_sample]
        normalized = self._normalize_for_replicates(cnt_positive, self.cnt_assays, self.gDNA_sample)
        
        # Store normalized gDNA counts
        self.tmm_norm[self.gDNA_sample] = normalized[self.gDNA_sample]
        
        # Calculate log fold changes for each assay
        self._calculate_cnt_logfc(normalized)
    
    def _calculate_cnt_logfc(self, normalized: pd.DataFrame) -> None:
        """
        Calculate log fold changes for ATAC and CUT&Tag.
        
        Args:
            normalized: Normalized count data
        """
        without_rep = self._get_index_without_replicate()
        
        for assay in self.cnt_assays:
            # Store normalized counts
            self.tmm_norm[assay] = normalized[assay]
            
            # Per-replicate log fold change
            logfc_rep = np.log2(normalized[assay] / normalized[self.gDNA_sample])
            self.logFC_reps[assay] = logfc_rep.replace([-np.inf, np.inf], np.nan)
            
            # Mean log fold change across replicates
            assay_sum = normalized[assay].groupby(without_rep).sum()
            gdna_sum = normalized[self.gDNA_sample].groupby(without_rep).sum()
            logfc_mean = np.log2(assay_sum / gdna_sum)
            self.logFC_mean[assay] = logfc_mean.replace([-np.inf, np.inf], np.nan)
    
    def _get_index_without_replicate(self) -> List[str]:
        """Get index columns excluding replicate column."""
        return [x for x in self.index_col if x != 'replicate']
    
    def _normalize_for_replicates(self, 
                                  data: pd.DataFrame, 
                                  assays: List[str], 
                                  gDNA: str) -> pd.DataFrame:
        """
        Apply TMM normalization for each replicate separately.
        
        Args:
            data: Input count data
            assays: List of assay column names
            gDNA: gDNA column name
            
        Returns:
            Normalized count data across all replicates
        """
        normalized_dfs = []
        
        for rep in self.reps:
            try:
                # Extract data for current replicate
                rep_data = data.xs(rep, level='replicate', drop_level=False)
                rep_subset = rep_data[assays + [gDNA]]
                
                # Calculate normalization factors
                norm_factors = self._calculate_tmm_factors(rep_subset, assays, gDNA)
                
                # Apply normalization
                size_factors = norm_factors * rep_subset.sum() / SCALING_FACTOR
                normalized_rep = rep_subset / size_factors
                
                normalized_dfs.append(normalized_rep)
                
            except KeyError:
                warnings.warn(f"Replicate {rep} not found in data")
                continue
        
        if not normalized_dfs:
            raise ValueError("No valid replicates found for normalization")
            
        return pd.concat(normalized_dfs)
    
    def _calculate_tmm_factors(self, 
                               data: pd.DataFrame, 
                               assays: List[str], 
                               gDNA_count: str,
                               tmm_ref: int = 1)-> pd.Series:
        """
        Calculate TMM (Trimmed Mean of M-values) normalization factors.
        
        This method implements the TMM normalization algorithm which is commonly used
        in RNA-seq data analysis to account for differences in library sizes.
        
        Args:
            data: Count data for normalization
            assays: List of assay columns to normalize
            gDNA_count: Reference gDNA column name
            tmm_ref: Reference feature index for TMM calculation
            
        Returns:
            Series of normalization factors for each assay
        """
        if gDNA_count == None:
            gDNA_count=self.gDNA_sample
        obs=data[assays]
        ref=data[gDNA_count]
        M=np.log2((obs/obs.sum()).div(ref/ref.sum(),axis='index'))
        A=np.log2((obs/obs.sum()).mul(ref/ref.sum(),axis='index'))/2
        M=M.loc[tmm_ref].replace(-np.inf,np.nan)
        A=A.loc[tmm_ref].replace(-np.inf,np.nan)
        
        length_M=M.rank().max()
        length_A=A.rank().max()
        mask_M=(0.3*length_M<M.rank())&(M.rank()<0.7*length_M)
        mask_A=(0.05*length_A<A.rank())&(A.rank()<0.95*length_A)
        mask=mask_M & mask_A
        
        f={}
        for column in obs.columns:
            j=obs.loc[tmm_ref][column][mask[column]]
            r=ref.loc[tmm_ref][mask[column]]
            
            w=(obs[column].sum()-j)/(obs[column].sum()*j)+(ref.sum()-r)/(ref.sum()*r)
            m=M[mask][column]
            f[column]=2**(np.sum(w*m)/w.sum())
        
        f=pd.Series(f)
        f[gDNA_count]=1

        return f