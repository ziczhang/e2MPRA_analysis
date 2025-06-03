import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.stats.multitest
from statannotations.Annotator import Annotator
sns.set()

DEFAULT_FEATURE_LABELS = {1: 'Rand. gen.',
                          2: 'Scram.',
                          3: 'Act. gen.',
                          4: 'Inact. gen.',
                          5: 'Act. syn.',
                          6: 'Inact. syn.'}

DEFAULT_COLORS = {1: '#F0E442',  # Yellow
                  2: '#E69F00',  # Orange
                  3: '#D55E00',  # Red-orange
                  4: '#56B4E9',  # Sky blue
                  5: '#009E73',  # Bluish green
                  6: '#CC79A7'   # Reddish purple
                 }  #Okabe-Ito palette

class CtlLibAnalyzer:
    """
    Analyzer for control library data with visualization and statistical testing capabilities.
    This class is designed for analysis and plotting method publication purposes.
    """
    
    def __init__(self, count_table, feature_labels=None, colors=None):
        """
        Initialize the analyzer with count table data and visualization settings.
        
        Args:
            count_table: Count table object with TMM normalization and logFC data
            feature_labels: Dictionary mapping feature IDs to readable labels (optional)
            colors: Dictionary mapping feature IDs to color codes for plots (optional)
        """
        
        # Store normalized data and logFC values
        self.tmm_norm = count_table.tmm_norm
        self.logFC_reps = count_table.logFC_reps
        self.logFC_mean = count_table.logFC_mean
        
        # Extract assay information and visualization settings
        self.assays = list(self.logFC_mean.columns)
        self.colors = colors if colors is not None else DEFAULT_COLORS
        self.feature_labels = feature_labels if feature_labels is not None else DEFAULT_FEATURE_LABELS
        self.features = self.logFC_mean.index.get_level_values('feature').unique()
        
    def load_endogenous_data(self, path):
        """
        Load and join endogenous data from external file.
        
        Args:
            path: File path to endogenous data (tab-separated format)
        """
        self.endogenous = pd.read_csv(path, sep='\t', index_col='id')
        self.logFC_mean = self.logFC_mean.join(self.endogenous)
    
    def plot_activity_distribution(self, ref=1):
        """
        Plot activity distribution with statistical comparisons.
        Creates violin plots for each assay with Mann-Whitney U tests.
        
        Args:
            ref: Reference feature ID for statistical comparisons (default: 1)
        """
        # Set up subplot layout
        n_assays = len(self.assays)
        fig, axes = plt.subplots(nrows=1, ncols=n_assays, 
                                 figsize=[3.5 * n_assays , 3.5], 
                                 tight_layout=True)
        
        # Ensure axes is always iterable (handle single subplot case)
        if n_assays == 1:
            axes = [axes]
        
        # Prepare data for plotting
        df = self._prepare_dataframe_for_plotting()
        
        # Define comparison pairs for statistical testing
        pairs = [(ref, other) for other in df['feature'].unique() if other != ref]
        
        # Create plots for each assay
        for ax, col in zip(axes, self.assays):
            self._create_violin_plot_with_statistics(ax, df, col, pairs)
            
    def plot_activity_correlation(self, col1, col2):
        """
        Plot correlation between two assays with Spearman correlation coefficient.
        
        Args:
            col1: First assay column name
            col2: Second assay column name
        """
        plt.figure(figsize=[4, 4], tight_layout=True)
        
        # Create scatter plot with feature-based coloring
        feature_values = self.logFC_mean.index.get_level_values('feature')
        sns.scatterplot(data=self.logFC_mean, 
                        x=col1, y=col2,
                        hue=feature_values, 
                        palette=self.colors)
        
        # Calculate and display Spearman correlation
        corr_matrix = self.logFC_mean[[col1, col2]].corr(method='spearman')
        correlation = corr_matrix.values[0, 1]
        
        plt.text(0.05, 0.95, 
                 f'ρ = {correlation:.2f}',
                 transform=plt.gca().transAxes, 
                 fontsize=12)
        
        # Remove legend and display plot
        plt.legend().remove()
        plt.show()
    
    def _prepare_dataframe_for_plotting(self):
        """
        Prepare dataframe for plotting by resetting index and adding color mapping.
        
        Returns:
            pd.DataFrame: Prepared dataframe with feature and color columns
        """
        df = self.logFC_mean.reset_index()
        df['color'] = df['feature'].map(self.colors)
        return df
    
    def _create_violin_plot_with_statistics(self, ax, df, column, pairs):
        """
        Create violin plot with statistical annotations for a single assay.
        
        Args:
            ax: Matplotlib axis object
            df: Plotting dataframe
            column: Column name for y-axis data
            pairs: List of feature pairs for statistical comparison
        """
        # Create violin plot
        sns.violinplot(data=df, x='feature', y=column, 
                       palette=self.colors, dodge=True, 
                       jitter=True, ax=ax)
        
        # Add statistical annotations
        annotator = Annotator(ax, pairs, data=df, x='feature', y=column)
        annotator.configure(test='Mann-Whitney', 
                            comparisons_correction='benjamini-hochberg',
                            text_format='star', loc='inside')
        
        annotator.apply_and_annotate()
        
        # Customize plot appearance
        ax.set_xticklabels(list(self.feature_labels.values()), 
                           rotation=90, ha='center')
        
        ax.set_ylabel('log2(Activity)')
        ax.set_xlabel(None)
        ax.set_title(column)