import re
import pandas as pd
import itertools
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import statsmodels.api as sm
import statsmodels.stats.multitest
from statsmodels.genmod.families import Gamma
from statsmodels.genmod.families.links import Log
from statannotations.Annotator import Annotator
from matplotlib.colors import LogNorm, Normalize

sns.set()

# Configuration Constants
NUCLEOTIDE_COLORS = {'A': '#E69F00', 'T': '#56B4E9', 'G': '#009E73', 'C': '#F0E442'}
NUCLEOTIDE_MARKERS = {'A': '^', 'T': '+', 'G': 'o', 'C': 'd'}
NUCLEOTIDE_HUE_ORDER = {'A': 0, 'T': 1, 'G': 2, 'C': 3}

# Statistical thresholds
P_VALUE_THRESHOLD = 0.01
MAD_MULTIPLIER = 1.4826

# Visualization parameters
BAR_WIDTH = 0.33
SIGNIFICANT_COLOR = '#555555'
NON_SIGNIFICANT_COLOR = '#c0c0c0'
MOTIF_LINE_WIDTH = 3
MOTIF_ALPHA = 0.5
SECONDARY_MOTIF_ALPHA = 0.2

# Peak detection parameters
DEFAULT_SIGMA = 2
DEFAULT_STRENGTH = 0.75
DEFAULT_LOWER_THRESHOLD = 50
NEIGHBOR_OFFSET_RANGE = [-1, 1]
MAX_CLUSTER_GAP = 1

# Default assay colors
DEFAULT_ASSAY_COLORS = {'H3K27ac': 'b', 'ATAC': 'y', 'lentiMPRA': 'r'}

# Plot dimensions
FIGURE_WIDTH = 16
ROW_HEIGHT = 3
MOTIF_ROW_HEIGHT = 0.5
Y_LIMIT_THRESHOLD = 2.0
DEFAULT_Y_LIMITS = [-2.5, 2.5]
Y_BUFFER = 0.5

# Position and coordinate settings
SEQUENCE_LENGTH = 100
MOTIF_Y_LIMIT = 5
POU5F1_SOX2_MOTIF = 'POU5F1::SOX2'


class WTC11LibAnalyzer:
    def __init__(self, count_table, motif_path, sample_label_path, 
                 use_samples=list(range(2, 10)), wt=[5, 6], 
                 single_nuc_substitutions=[7], window_perturbation=[8, 9]):
        """
        Initialize the WTC11 library analyzer with count data and sample information.
        
        Args:
            count_table: Count table object containing expression data
            motif_path: Path to motif location file
            sample_label_path: Path to sample label file
            use_samples: List of sample indices to analyze
            wt: Wild-type sample indices
            single_nuc_substitutions: Single nucleotide substitution sample indices
            window_perturbation: Window perturbation sample indices
        """
        if not count_table.processed:
            count_table.process()
            
        self.tmm_norm = count_table.tmm_norm
        self.logFC_reps = count_table.logFC_reps
        self.logFC_mean = count_table.logFC_mean
        self.assays = list(self.logFC_mean.columns)

        self.wt = wt
        self.mut_sample = single_nuc_substitutions
        self.perturb_sample = window_perturbation
        self.samples = use_samples

        self._load_additional_data(motif_path, sample_label_path)
        self._process_analysis_data()
        
    def _load_additional_data(self, motif_path, sample_label_path):
        """Load motif locations and sample information from files."""
        self.motif_location = pd.read_csv(motif_path, delimiter='\t', header=0, index_col='target')
        self.sample_id = pd.read_csv(sample_label_path, delimiter='\t', header=0, index_col='sample')
        
    def _process_analysis_data(self):
        """Process regression analysis and estimate wild-type activity."""
        self.regression_res = self._perform_mutagenesis_regression()
        self.wt_act = self._estimate_wildtype_activity()
        
    def plot_single_nucleotide_substitution_effects(self, assays=None, samples=None):
        """
        Plot the effects of single nucleotide substitutions across genomic positions.
        
        Args:
            assays: List of assays to plot (default: all assays)
            samples: List of samples to plot (default: all samples)
        """
        if assays is None:
            assays = self.assays
        if samples is None:
            samples = self.samples
        
        for sample in samples:
            fig, axes = plt.subplots(
                nrows=len(assays) + 1, ncols=1, 
                figsize=[FIGURE_WIDTH, ROW_HEIGHT * (len(assays) + MOTIF_ROW_HEIGHT)],
                gridspec_kw={'height_ratios': [MOTIF_ROW_HEIGHT] + [1] * len(assays)},
                sharex=True
            )

            self._add_motif_annotation(axes[0], sample)

            for ax, assay in zip(axes[1:], assays):
                df = self.regression_res.query('sample==@sample & assay==@assay')
                self._plot_variant_effects(df, ax)
                self._decorate_substitution_plot(ax, assay, sample)
                ax.set_ylabel('Variant Effect')

            plt.show()
    
    def _plot_variant_effects(self, df, ax):
        """Plot variant effects as bars with significance coloring and nucleotide markers."""
        threshold = -np.log10(P_VALUE_THRESHOLD)
        
        for idx, row in df.iterrows():
            x = row['position'] + BAR_WIDTH * NUCLEOTIDE_HUE_ORDER[row['mutation']] - 1
            
            # Color based on statistical significance
            color = SIGNIFICANT_COLOR if row['-log10(p)'] > threshold else NON_SIGNIFICANT_COLOR
            
            ax.bar(x, row['coef'], width=BAR_WIDTH, linewidth=0, color=color)
            ax.scatter(x, row['coef'], label=row['mutation'], 
                      marker=NUCLEOTIDE_MARKERS[row['mutation']], 
                      c=NUCLEOTIDE_COLORS[row['mutation']])

    def _decorate_substitution_plot(self, ax, assay, sample):
        """Add decorations and labels to substitution effect plots."""
        ax.text(0, 1.01, 'WT log2(Activity)=%.2f' % (self.wt_act.at[sample, assay]),
                transform=ax.transAxes, fontsize=12)
        ax.set_title(assay)
        ax.set_xlim([-0.5, SEQUENCE_LENGTH + 0.5])
            
    def _add_motif_annotation(self, ax, sample):
        """Add motif location annotations to the top subplot."""
        seq_name = self.sample_id.at[sample, 'seq_name']
        coordinate = self.sample_id.at[sample, 'coordinate']
        locations = self.motif_location.loc[seq_name]
        
        s_past, e_past = 0, 0
        
        for _, row in locations.iterrows():
            start = row['start']
            end = row['end']
            motif = row['TF']

            if motif == POU5F1_SOX2_MOTIF:
                ax.plot([start, end], [0, 0], lw=MOTIF_LINE_WIDTH, 
                       alpha=MOTIF_ALPHA, color='black')
                ax.annotate("POU5F1::SOX2", xy=((start + end) / 2, 0.1), ha='center')
            else:
                # Prevent annotation overlaps
                if start < e_past:
                    y += 1
                else:
                    y = 1
                ax.plot([start, end], [y, y], lw=MOTIF_LINE_WIDTH, alpha=SECONDARY_MOTIF_ALPHA)
                ax.annotate(motif, xy=((start + end) / 2, y + 0.1), ha='center')
                s_past, e_past = start, end

        ax.set_ylim(0, MOTIF_Y_LIMIT)
        ax.set_yticks([])
        ax.set_facecolor('white')
        ax.set_title(f'{seq_name} {coordinate}', loc="left")

    def _perform_mutagenesis_regression(self):
        """Perform regression analysis for mutagenesis effects."""
        results = []
        
        for sample in self.samples:
            pattern = self._create_mutation_design_matrix(sample)
            
            for assay in self.assays:
                X, y = self._prepare_regression_data(pattern, assay, sample)
                X = sm.add_constant(X)
                
                # Filter out non-finite values
                mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
                
                model = self._fit_regression_model(X[mask], y[mask])
                df = self._extract_regression_results(model, sample, assay)
                results.append(df)

        return pd.concat(results)

    def _create_mutation_design_matrix(self, sample):
        """Create design matrix for mutation effects."""
        data = self.tmm_norm.loc[self.wt + self.mut_sample, sample, :].sort_index()
        
        # Create dummy variables for mutations
        mutation_matrix = pd.get_dummies(
            data.index.get_level_values('id').str.extract("([ATGC]+\d+[ATGC]+)$")[0]
        )
        
        # Add replicate effects
        replicate_dummies = pd.get_dummies(
            data.index.get_level_values('replicate'), drop_first=True
        )
        mutation_matrix[['rep2', 'rep3']] = replicate_dummies
        mutation_matrix.index = data.index
        
        return mutation_matrix

    def _prepare_regression_data(self, pattern, assay, sample):
        """Prepare X and y data for regression based on assay type."""
        if assay == 'lentiMPRA':
            X = pattern.copy()
            X['log2dna'] = np.log2(self.tmm_norm['dna_count'])
            y = np.log2(self.tmm_norm.loc[self.wt + self.mut_sample, sample, :]['rna_count'])
        else:
            X = pattern.loc[self.mut_sample].copy()
            X['log2dna'] = np.log2(self.tmm_norm['gDNA'])
            y = np.log2(self.tmm_norm.loc[self.mut_sample, sample, :][assay])
            
        return X, y
        
    def _fit_regression_model(self, X, y):
        """Fit OLS regression model."""
        return sm.OLS(y, X).fit()

    def _extract_regression_results(self, model, sample, assay):
        """Extract and format regression results."""
        df = pd.DataFrame({
            'coef': model.params,
            '-log10(p)': -np.log10(model.pvalues)
        }).drop(['const', 'rep2', 'rep3', 'log2dna'])
        
        # Parse mutation information
        df[['original', 'position', 'mutation']] = df.index.str.extract("([ATGC]+)(\d+)([ATGC]+)$").values
        df['position'] = df['position'].astype(int)
        df['sample'] = sample
        df['assay'] = assay
        
        return df

    def plot_perturbation_effects(self, assays=None, samples=None, colors=None):
        """
        Plot the effects of window perturbations using MAD scores.
        
        Args:
            assays: List of assays to plot
            samples: List of samples to plot
            colors: Dictionary mapping assay names to colors
        """
        if assays is None:
            assays = self.assays
        if samples is None:
            samples = self.samples
        if colors is None:
            colors = DEFAULT_ASSAY_COLORS

        perturbation_matrix = self._create_perturbation_matrix()
        
        for sample in samples:
            fig, axes = plt.subplots(
                nrows=len(assays) + 1, ncols=1, 
                figsize=[FIGURE_WIDTH, ROW_HEIGHT * (len(assays) + MOTIF_ROW_HEIGHT)],
                gridspec_kw={'height_ratios': [MOTIF_ROW_HEIGHT] + [1] * len(assays)},
                sharex=True
            )

            self._add_motif_annotation(axes[0], sample)

            for ax, assay in zip(axes[1:], assays):
                mad_scores = self._calculate_positional_mad_scores(perturbation_matrix, assay, sample)
                wt_activity = self.wt_act.at[sample, assay]
                
                smoothed_signal, functional_ranges, edge_clusters, peaks = self.identify_functional_sites(
                    mad_scores, wt_activity, sigma=DEFAULT_SIGMA, 
                    strength=DEFAULT_STRENGTH, lower_threshold=DEFAULT_LOWER_THRESHOLD
                )

                x = mad_scores.index.to_numpy()

                ax.bar(x, mad_scores, color=colors[assay])
                ax.plot(x, smoothed_signal, linewidth=3, color=colors[assay])
                
                # Highlight functional sites
                for start, end in functional_ranges:
                    ax.hlines(y=0, xmin=start, xmax=end, color="gray", 
                             linestyle="solid", linewidth=8)
                
                self._decorate_perturbation_plot(ax, assay, sample, mad_scores)

            plt.show()

    def _estimate_wildtype_activity(self):
        """Estimate wild-type activity levels across samples and assays."""
        mpra_activity = self.logFC_mean.loc[self.wt, 'lentiMPRA'].reset_index(['feature', 'id'], drop=True)
        other_activities = self.logFC_reps.loc[self.mut_sample].groupby('sample').median().drop('lentiMPRA', axis=1)
        
        return pd.concat([mpra_activity, other_activities], axis=1)

    def _create_perturbation_matrix(self):
        """Create matrix indicating perturbation locations."""
        idx = self.logFC_mean.loc[self.perturb_sample].index
        id_list = idx.get_level_values('id')
        
        pattern = pd.DataFrame(
            np.zeros([len(id_list), SEQUENCE_LENGTH]),
            index=idx, 
            columns=list(range(0, SEQUENCE_LENGTH))
        )
        pattern = pattern.replace(0, np.nan)

        # Extract perturbation coordinates
        location = id_list.str.extract("(\d+)-(\d+)perturbated")
        location.index = idx
        
        for row in location.itertuples():
            start_pos = int(row[1]) - 1
            end_pos = int(row[2]) - 1
            pattern.loc[row[0], start_pos:end_pos] = 1

        return pattern

    def _calculate_positional_mad_scores(self, matrix, assay, sample):
        """Calculate MAD scores for positional effects."""
        wt_activity = self.wt_act.at[sample, assay]
        
        activity = self.logFC_mean.loc[self.perturb_sample, sample, :][assay]
        pattern = matrix.loc[self.perturb_sample, sample, :]

        positional_activity = pattern.mul(activity, axis='index').median(axis=0).dropna()
        
        # Calculate MAD score
        activity_median = np.median(np.abs(activity.dropna() - wt_activity))
        mad_score = (positional_activity - wt_activity) / (activity_median * MAD_MULTIPLIER)

        return mad_score

    def _decorate_perturbation_plot(self, ax, assay, sample, mad_scores):
        """Add decorations to perturbation effect plots."""
        ax.text(0, 1.01, 'WT log2(Activity)=%.2f' % (self.wt_act.at[sample, assay]),
                transform=ax.transAxes, fontsize=12)
        ax.set_title(assay)
        ax.set_ylabel('MAD Score')
        ax.set_xlim([-0.5, SEQUENCE_LENGTH + 0.5])
        
        # Set appropriate y-limits
        if np.abs(mad_scores).max() > Y_LIMIT_THRESHOLD:
            ax.set_ylim([mad_scores.min() - Y_BUFFER, mad_scores.max() + Y_BUFFER])
        else:
            ax.set_ylim(DEFAULT_Y_LIMITS)

    def _split_clusters_at_zero_crossing(self, cluster, signal):
        """Split clusters at zero crossings in the signal."""
        new_clusters = []
        temp_cluster = []
    
        for i in range(len(cluster) - 1):
            temp_cluster.append(cluster[i])
            if signal[cluster[i]] * signal[cluster[i + 1]] < 0:
                new_clusters.append(temp_cluster)
                temp_cluster = []
        
        temp_cluster.append(cluster[-1])
        if temp_cluster:
            new_clusters.append(temp_cluster)
        
        return new_clusters

    def _detect_edge_clusters_canny_style(self, gradient, lower_threshold_percentile):
        """Detect edge clusters using Canny-style edge detection."""
        threshold_low = np.percentile(np.abs(gradient), lower_threshold_percentile)
        
        # Find gradient extrema (peaks and valleys)
        gradient_extrema = set(np.append(find_peaks(gradient)[0], find_peaks(-gradient)[0]))
        weak_edges = set(np.where((np.abs(gradient) >= threshold_low))[0])
        strong_edges = gradient_extrema & weak_edges
        
        all_edges = set(strong_edges)
        
        def expand_edge_clusters(strong, weak):
            """Recursively expand edge clusters by including neighboring weak edges."""
            newly_added = set()
            for strong_edge in list(strong):
                for offset in NEIGHBOR_OFFSET_RANGE:
                    neighbor = strong_edge + offset
                    if neighbor in weak:
                        newly_added.add(neighbor)
            
            if newly_added:
                weak -= newly_added
                strong |= newly_added
                return expand_edge_clusters(strong, weak)
            return strong
        
        all_edges = expand_edge_clusters(all_edges, weak_edges)
    
        # Handle isolated edges
        isolated_edges = strong_edges - set(all_edges)
        for edge in isolated_edges:
            all_edges.append([edge])
        
        # Group edges into clusters
        sorted_edges = sorted(list(all_edges))
        clusters = []
        current_cluster = []
        
        for i in range(len(sorted_edges)):
            if not current_cluster or sorted_edges[i] <= current_cluster[-1] + MAX_CLUSTER_GAP:
                current_cluster.append(sorted_edges[i])
            else:
                clusters.append(current_cluster)
                current_cluster = [sorted_edges[i]]
        
        if current_cluster:
            clusters.append(current_cluster)
          
        return clusters

    def _identify_valid_peak_ranges(self, peaks, valleys, all_edges, gradient, smoothed_signal):
        """Identify valid peak ranges based on edge clusters and gradient direction."""
        valid_peak_ranges = []
        
        # Process peaks
        for peak in peaks:
            positive_edges = [cluster for cluster in all_edges if smoothed_signal[min(cluster)] > 0]
            
            left_edge = max([cluster for cluster in positive_edges if max(cluster) < peak], 
                           default=None, key=max)
            right_edge = min([cluster for cluster in positive_edges if min(cluster) > peak], 
                            default=None, key=min)
            
            if left_edge is not None and gradient[min(left_edge)] > 0:
                valid_peak_ranges.append((min(left_edge), peak))
            if right_edge is not None and gradient[max(right_edge)] < 0:
                valid_peak_ranges.append((peak, max(right_edge)))
                
        # Process valleys
        for valley in valleys:
            negative_edges = [cluster for cluster in all_edges if smoothed_signal[min(cluster)] < 0]
            
            left_edge = max([cluster for cluster in negative_edges if max(cluster) < valley], 
                           default=None, key=max)
            right_edge = min([cluster for cluster in negative_edges if min(cluster) > valley], 
                            default=None, key=min)
            
            if left_edge is not None and gradient[min(left_edge)] < 0:
                valid_peak_ranges.append((min(left_edge), valley))
            if right_edge is not None and gradient[max(right_edge)] > 0:
                valid_peak_ranges.append((valley, max(right_edge)))
        
        return valid_peak_ranges

    def identify_functional_sites(self, positional_activity, baseline_activity, 
                                 sigma=DEFAULT_SIGMA, strength=DEFAULT_STRENGTH, 
                                 lower_threshold=DEFAULT_LOWER_THRESHOLD):
        """
        Identify functional sites using smoothing, peak detection, and edge clustering.
        
        Args:
            positional_activity: Series of positional activity values
            baseline_activity: Baseline activity level
            sigma: Gaussian smoothing parameter
            strength: Minimum peak/valley strength threshold
            lower_threshold: Lower threshold percentile for edge detection
            
        Returns:
            Tuple of (smoothed_signal, valid_peak_ranges, edge_clusters, all_peaks_valleys)
        """
        x = positional_activity.index.to_numpy()
        smoothed_signal = gaussian_filter1d(positional_activity, sigma=sigma)
        
        gradient = np.gradient(smoothed_signal)
        edge_clusters = self._detect_edge_clusters_canny_style(gradient, lower_threshold)
        
        # Split edge clusters at zero crossings
        split_edge_clusters = []
        for cluster in edge_clusters:
            split_edge_clusters.extend(self._split_clusters_at_zero_crossing(cluster, smoothed_signal))
        
        all_edges = split_edge_clusters  
        
        # Find significant peaks and valleys
        peaks, _ = find_peaks(smoothed_signal)
        peaks = peaks[np.abs(smoothed_signal[peaks]) > strength]
        
        valleys, _ = find_peaks(-smoothed_signal)
        valleys = valleys[np.abs(smoothed_signal[valleys]) > strength]
        
        valid_peak_ranges = self._identify_valid_peak_ranges(peaks, valleys, all_edges, gradient, smoothed_signal)
        
        return smoothed_signal, valid_peak_ranges, edge_clusters, np.append(peaks, valleys)