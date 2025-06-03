import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from scipy.special import comb
from scipy.stats import spearmanr
from scipy.stats import gaussian_kde
import itertools
from typing import List, Optional, Tuple, Any

sns.set()

# Configuration constants
DEFAULT_BINS = 20
FIGURE_WIDTH_PER_SUBPLOT = 3.0
FIGURE_HEIGHT_PER_SUBPLOT = 3.0
SCATTER_POINT_SIZE = 5.0
COLORMAP = 'gnuplot2'
FONT_SIZE = 12
MAX_TICKS = 6
CORRELATION_TEXT_X = 0.05
CORRELATION_TEXT_Y = 0.9

class QuarityChecker:
    """Quality checker for genomic data analysis and visualization"""
    
    def __init__(self, count_table):
        """Initialize QuarityChecker with count table
        
        Args:
            count_table: Count table object containing the data to analyze
        """
        if not count_table.processed:
            count_table.process()

        self.ct = count_table
    
    def _validate_mpra_data(self) -> None:
        """Validate MPRA data availability"""
        if self.ct.mpra_ori is None:
            raise ValueError('The lentiMPRA result is empty')
    
    def _validate_count_data(self) -> None:
        """Validate ATAC and CUT&Tag count data availability"""
        if self.ct.cnt_ori is None:
            raise ValueError('The ATAC and CUT&Tag result is empty')
    
    def _decorate(self, ax, xlabel, logx: bool = False, logy: bool = False) -> None:
        """Configure logarithmic scales for axes
        
        Args:
            ax: Matplotlib axis object
            xlabel: xlabel
            logx: Whether to apply log scale to x-axis
            logy: Whether to apply log scale to y-axis
        """
        if logy:
            ax.set_ylim(bottom=1)
            ax.set_yscale('log')

        if logx:
            xlabel = f'log10({xlabel} + 1)'
        else:
            xlabel = xlabel

        ax.set_xlabel(xlabel)
        
        # ax.set_xlim(left=-0.1)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, prune='lower', nbins=MAX_TICKS))
    
    def _compute_kde_density(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute kernel density estimation for scatter plot coloring
        
        Args:
            x: X coordinates
            y: Y coordinates
            
        Returns:
            Density values for each point
        """
        xy = np.vstack([x, y])
        return gaussian_kde(xy)(xy)
    
    def _add_correlation_text(self, ax, correlation: float) -> None:
        """Add correlation coefficient text to plot
        
        Args:
            ax: Matplotlib axis object
            correlation: Correlation coefficient value
        """
        ax.text(CORRELATION_TEXT_X, 
                CORRELATION_TEXT_Y, 
                f'ρ = {correlation:.2f}',
                transform=ax.transAxes, 
                fontsize=FONT_SIZE)
    
    def _generate_axis_labels(self, label: Optional[str], log: bool, 
                              pair: Tuple[Any, Any]) -> Tuple[str, str]:
        """Generate appropriate axis labels for correlation plots
        
        Args:
            label: Optional label for the data type
            log: Whether log transformation is applied
            pair: Tuple of replicate identifiers
            
        Returns:
            Tuple of (xlabel, ylabel)
        """
        if label:
            if log:
                xlabel = f'log10({label}+1) rep{pair[0]}'
                ylabel = f'log10({label}+1) rep{pair[1]}'
            else:
                xlabel = f'{label} rep{pair[0]}'
                ylabel = f'{label} rep{pair[1]}'
        else:
            if log:
                xlabel = f'log10(value+1) rep{pair[0]}'
                ylabel = f'log10(value+1) rep{pair[1]}'
            else:
                xlabel = f'rep{pair[0]}'
                ylabel = f'rep{pair[1]}'
        
        return xlabel, ylabel

    def _histogram_for_reps(self, axes, df, title, xlabel, logx, logy):
        
        for ax, rep in zip(axes, self.ct.reps):
            data = df[rep]
            
            if logx:
                plot_data = np.log10(data + 1)
            else:
                plot_data = data
            
            sns.histplot(plot_data, ax=ax, bins=DEFAULT_BINS)

            ax.set_title(f'{title} rep{rep}')
            
            self._decorate(ax, xlabel, logx, logy)
        
    def plot_barcode_coverage_distribution(self, logx: bool = False, logy: bool = False) -> None:
        """Plot distribution of barcode coverage per CRE across replicates
        
        Args:
            logx: Whether to use log scale for x-axis
            logy: Whether to use log scale for y-axis
        """
        self._validate_mpra_data()
        
        df = self.ct.mpra_ori.pivot_table(index='id', 
                                          columns='replicate', 
                                          values='n_obs_bc')
        
        n_reps = len(self.ct.reps)
        fig, axes = plt.subplots(ncols=n_reps, 
                                 nrows=1, 
                                 figsize=[FIGURE_WIDTH_PER_SUBPLOT * n_reps, 4],
                                 tight_layout=True, 
                                 sharex=True, 
                                 sharey=True)

        title='n_bos_bc'
        xlabel='#BCs per CRE'

        self._histogram_for_reps(axes, df, title, xlabel, logx, logy)

        plt.show()

    def plot_umi_count_distribution(self, logx: bool = False, logy: bool = False) -> None:
        """Plot distribution of UMI counts across samples and replicates
        
        Args:
            logx: Whether to use log scale for x-axis
            logy: Whether to use log scale for y-axis
        """
        self._validate_count_data()
        
        samples = [self.ct.gDNA_sample] + self.ct.cnt_assays
        ncols = len(self.ct.reps)
        nrows = len(samples)
        
        fig, axes = plt.subplots(ncols=ncols, 
                                 nrows=nrows,
                                 figsize=[FIGURE_WIDTH_PER_SUBPLOT * ncols, 
                                          FIGURE_HEIGHT_PER_SUBPLOT * nrows],
                                 tight_layout=True, 
                                 sharex=True, 
                                 sharey=True)
        
        for i, sample in enumerate(samples):

            df = self.ct.cnt_ori.pivot_table(index='id', 
                                             columns='replicate', 
                                             values=sample)
            title=sample
            xlabel='#UMIs per CRE'
            
            self._histogram_for_reps(axes[i,:], df, title, xlabel, logx, logy)
        
        plt.show()

    def plot_umi_correlation(self, log: bool = False) -> None:
        """Plot UMI count correlations between replicates
        
        Args:
            log: Whether to apply log transformation
        """
        self._validate_count_data()
        samples = [self.ct.gDNA_sample] + self.ct.cnt_assays
        self._replicate_correlation_scatter(self.ct.cnt_ori, samples, log=log, label='#UMIs')
        
    def plot_logFC_correlation(self) -> None:
        """Plot log fold-change correlations between replicates"""
        self._replicate_correlation_scatter(self.ct.logFC_reps, self.ct.logFC_reps.columns)

    def plot_activity_correlation(self, keys: List[int] = [1, 2, 3, 4]) -> None:
        """Plot activity distribution across sequence features
        
        Args:
            keys: List of feature keys to include in the plot
        """
        n_cols = len(self.ct.logFC_mean.columns)
        fig, axes = plt.subplots(nrows=1, 
                                 ncols=n_cols,
                                 figsize=[FIGURE_WIDTH_PER_SUBPLOT * n_cols, 3],
                                 sharey=False, 
                                 tight_layout=True)
        
        axes = np.atleast_1d(axes)
        data = self.ct.logFC_mean.loc[keys].reset_index()
        
        for ax, col in zip(axes, self.ct.logFC_mean.columns):
            sns.violinplot(data=data, x='feature', y=col, ax=ax)
            ax.set_xlabel('seq feature')
            ax.set_ylabel('log2(Activity)')
            ax.set_xticklabels(keys)
            ax.set_title(col)
        
        plt.show()

    def plot_activity_correlation(self) -> None:
        """Plot activity correlations between different conditions"""
        df = self.ct.logFC_mean
        samples = self.ct.logFC_mean.columns
        pairs = list(itertools.combinations(samples, 2))
        ncols = len(pairs)
        
        fig, axes = plt.subplots(ncols=ncols, 
                                 nrows=1,
                                 figsize=[4 * ncols, 4], 
                                 tight_layout=True)
        
        for ax, pair in zip(axes, pairs):
            sns.scatterplot(x=self.ct.logFC_mean[pair[1]], 
                            y=self.ct.logFC_mean[pair[0]], 
                            ax=ax)
            
            correlation = self.ct.logFC_mean[list(pair)].corr(method='spearman').values[0, 1]
            
            self._add_correlation_text(ax, correlation)
        
        plt.show()
            
    def _replicate_correlation_scatter(self, data, samples, log: bool = False, 
                                       label: Optional[str] = None) -> None:
        """Generic method for plotting correlations between replicates
        
        Args:
            data: Data to analyze
            samples: List of sample names
            log: Whether to apply log transformation
            label: Optional label for axis labeling
        """
        df_pivot = data.pivot_table(index='id', 
                                    columns='replicate', 
                                    values=samples)

        n_combinations = comb(len(self.ct.reps), 2, exact=True)
        nrows = len(samples)
        
        fig, axes = plt.subplots(ncols=n_combinations, 
                                 nrows=nrows,
                                 figsize=[FIGURE_WIDTH_PER_SUBPLOT * n_combinations, 
                                          FIGURE_HEIGHT_PER_SUBPLOT * nrows],
                                 tight_layout=True)
        
        axes = np.array(axes).reshape(nrows, n_combinations)

        for i, sample in enumerate(samples):
            for j, pair in enumerate(itertools.combinations(self.ct.reps, 2)):
                ax = axes[i, j]
                
                # Prepare data for correlation analysis
                df_sample = df_pivot[sample][list(pair)].dropna()
                correlation = df_sample.corr(method='spearman').values[0, 1]
                
                x_data = df_sample[pair[0]]
                y_data = df_sample[pair[1]]
                
                # Apply log transformation if requested
                if log:
                    x_data = np.log10(x_data + 1)
                    y_data = np.log10(y_data + 1)
                
                # Create scatter plot with density coloring
                density_colors = self._compute_kde_density(x_data, y_data)
                
                self._add_correlation_text(ax, correlation)
                sns.scatterplot(x=x_data, 
                                y=y_data, 
                                c=density_colors,
                                cmap=COLORMAP, 
                                ax=ax, 
                                s=SCATTER_POINT_SIZE)
                
                ax.set_title(sample)

                # Set axis labels based on parameters
                xlabel, ylabel = self._generate_axis_labels(label, log, pair)
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                    
        plt.show()