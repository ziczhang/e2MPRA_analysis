import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm, Normalize
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import seaborn as sns
import scipy.stats as stats
from scipy.special import perm
from scipy.stats import hypergeom
import statsmodels.api as sm
import statsmodels.stats.multitest
from statannotations.Annotator import Annotator
import itertools
import networkx as nx
from collections import Counter

# Configuration Constants
DEFAULT_MOTIFS = ['CEBPA', 'CTCF', 'FOXA1', 'HNF1A', 'ONECUT1', 'PPARA', 'XBP1', 'REST']
DEFAULT_TEMPLATE_LABELS = [5, 6]
DEFAULT_CLASS1_LABELS = [7, 8]
DEFAULT_CLASS2_LABELS = [9, 10]
DEFAULT_CLASS3_LABELS = [11, 12]

# Color palettes
MOTIF_COLORS = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7', '#000000']
ASSAY_COLORS = {'H3K27ac': 'b', 'ATAC': 'r', 'lentiMPRA': 'c'}

# Statistical thresholds
FDR_ALPHA = 0.01
STRICT_FDR_ALPHA = 0.01
SIGNIFICANCE_THRESHOLD = 0.01

# Plotting parameters
DEFAULT_PLOT_YLIM = (-3, 3)
DEFAULT_BOXPLOT_SIZE = (2, 2)
NETWORK_PLOT_SIZE = (6, 6)
PERMUTATION_PLOT_SIZE = (8, 6)
POSITION_ENRICHMENT_SIZE = 200
HYPERGEO_TOTAL_POPULATION = 1680
HYPERGEO_SUCCESS_STATES = 210

# Styling
sns.set()
plt.style.use('tableau-colorblind10')


class HepG2LibAnalyzer:
    def __init__(self, count_table, matrix_path,
                 template_label=None, class1_label=None, class2_label=None, class3_label=None):
        """
        Initialize the HepG2 library analyzer.
        
        Args:
            count_table: Count table object with processed data
            matrix_path: Path to the pattern matrix file
            template_label: Template feature labels
            class1_label: Class 1 feature labels
            class2_label: Class 2 feature labels
            class3_label: Class 3 feature labels
        """
        # Initialize count table data
        if not count_table.processed:
            count_table.process()
        self.tmm_norm = count_table.tmm_norm
        self.logFC_reps = count_table.logFC_reps
        self.logFC_mean = count_table.logFC_mean
        self.assays = list(self.logFC_mean.columns)

        # Set feature labels with defaults
        self.template = template_label or DEFAULT_TEMPLATE_LABELS
        self.class1 = class1_label or DEFAULT_CLASS1_LABELS
        self.class2 = class2_label or DEFAULT_CLASS2_LABELS
        self.class3 = class3_label or DEFAULT_CLASS3_LABELS

        # Initialize motifs and colors
        self.motifs = DEFAULT_MOTIFS
        self._setup_color_palette()

        # Load and process data
        self._load_pattern_matrix(matrix_path)
        self._process_class1_correlation_data()
        self._process_class2_interaction_data()
        self._process_class3_permutation_data()

    def _setup_color_palette(self):
        """Setup color palette for motifs."""
        if len(self.motifs) <= len(MOTIF_COLORS):
            self.colors = MOTIF_COLORS[:len(self.motifs)]
        else:
            rgb_palette = sns.color_palette("husl", len(self.motifs))
            self.colors = ['#{:02x}{:02x}{:02x}'.format(int(r*255), int(g*255), int(b*255)) 
                          for r, g, b in rgb_palette]

    def _load_pattern_matrix(self, path):
        """Load pattern matrix from file."""
        self.pattern_table = pd.read_csv(path, delimiter='\t', header=0, 
                                       index_col=['feature', 'id'])

    def _process_class1_correlation_data(self, motifs=None, assays=None):
        """Process class 1 data for correlation analysis between motif counts and activity."""
        if motifs is None:
            motifs = self.motifs
        if assays is None:
            assays = self.assays
        
        logFC = self.logFC_reps.loc[self.class1]
        nsite = self.pattern_table.loc[self.class1, motifs]

        data = logFC.join(nsite)
        results = []

        for assay in assays:
            for motif in motifs:
                df = data[[motif, assay]]
                df = df[df[motif] != 0].dropna()
                
                if len(df) > 1:  # Need at least 2 points for correlation
                    rho, p = stats.spearmanr(df[[motif, assay]])
                    results.append([motif, assay, rho, p, df[motif].values, df[assay].values])

        results_df = pd.DataFrame(results, columns=['motif', 'assay', 'rho', 'p', 'nsites', 'value']).dropna()
        results_df['sig'], results_df['adjp'] = statsmodels.stats.multitest.fdrcorrection(
            results_df['p'], is_sorted=False, alpha=0.05)

        self.class1_correlation_data = results_df

    def _process_class2_interaction_data(self):
        """Process class 2 data for motif interaction analysis."""
        self._merge_class2_data()
        self.class2_regression_results = self._perform_class2_regression()

    def _process_class3_permutation_data(self):
        """Process class 3 data for permutation analysis."""
        self.class3_permutation_data = self.logFC_reps.join(self.pattern_table).loc[self.class3]

        # Extract permutation information from IDs
        perm = self.class3_permutation_data.index.get_level_values('id').str.replace(
            'template\d_class3_', '', regex=True)
        elements = perm.str.split('_')
        is_valid = elements.map(lambda lst: all(item in set(self.motifs) for item in lst))
        
        self.class3_permutation_data = self.class3_permutation_data.loc[is_valid]
        self.class3_permutation_data.index = perm[is_valid]

        self.permutation_anova_results = self._perform_permutation_anova()
   
    def plot_class1_motif_site_correlation(self, use_motifs=None, use_assays=None):
        """Plot correlation between number of motif sites and activity for class 1."""
        if use_motifs is None:
            use_motifs = self.motifs
        if use_assays is None:
            use_assays = self.assays

        fig, axes = plt.subplots(ncols=len(use_motifs), nrows=len(use_assays), 
                               sharex=True, figsize=[2*len(use_motifs), 2*len(use_assays)], 
                               tight_layout=True)
        
        # Prepare data for plotting
        plot_data = self.class1_correlation_data[['motif', 'assay', 'nsites', 'value']].apply(pd.Series.explode)
        
        self._create_correlation_boxplots(axes, plot_data, 
                                        {'motif': use_motifs}, {'assay': use_assays}, 
                                        'nsites', 'value')

        # Add significance annotations and formatting
        for i, assay in enumerate(use_assays):
            for j, motif in enumerate(use_motifs):
                ax = axes[i, j]
                df = self.class1_correlation_data.query(f'motif==@motif & assay==@assay')
                if not df.empty:
                    self._add_significance_annotation(ax, df['adjp'].values[0], sig=df['sig'].values[0])

                # Set labels and limits
                if i == 0:
                    ax.set_title(motif)
                if j == 0:
                    ax.set_ylabel(f'{assay}\nlog2(Activity)')
                else:
                    ax.set_ylabel(None)

                ax.set_xlabel(None)
                ax.set_ylim(*DEFAULT_PLOT_YLIM)

        # Set x-axis labels for bottom row
        for ax in axes[-1, :]:
            ax.set_xlabel('Number of motifs')

        plt.show()     

    def _create_correlation_boxplots(self, axes, data, cols: dict, rows: dict, x, y, colors=None):
        """Create boxplots for correlation analysis."""
        if colors is None:
            colors = ASSAY_COLORS
            
        col_key = list(cols.keys())[0]
        col_values = list(cols.values())[0]
        row_key = list(rows.keys())[0]
        row_values = list(rows.values())[0]

        for i, row in enumerate(row_values):
            for j, col in enumerate(col_values):
                df = data.query(f'{row_key}==@row & {col_key}==@col')
                
                ax = axes[i, j]
                if colors and row in colors:
                    sns.boxplot(data=df, x=x, y=y, ax=ax, color=colors[row])
                else:
                    sns.boxplot(data=df, x=x, y=y, ax=ax)
                    
                sns.swarmplot(data=df, x=x, y=y, ax=ax, color='black')

    def _add_significance_annotation(self, ax, adjp, sig=None, alpha=SIGNIFICANCE_THRESHOLD):
        """Add significance annotation to plot."""
        if sig is None:
            sig = adjp < alpha
        
        ax.text(0.05, 0.95, f'adj-p = {adjp:.3f}', transform=ax.transAxes, fontsize=10)

        if sig:
            ax.set_facecolor((1.0, 0.5, 0.5, 0.3))  # Light red for significant
        else:
            ax.set_facecolor((0.5, 0.7, 1.0, 0.2))  # Light blue for non-significant

    def plot_class2_interaction_boxplots(self, use_motifs1=None, use_motifs2=None, use_assays=None, 
                                       use_class1_n=4, use_class2_n=2):
        """Plot boxplots for class 2 motif interactions."""
        if use_motifs1 is None:
            use_motifs1 = self.motifs
        if use_motifs2 is None:
            use_motifs2 = self.motifs
        if use_assays is None:
            use_assays = self.assays
    
        data_dict = {}
    
        for assay in use_assays:
            assay_data = {}
            for i, site1 in enumerate(use_motifs1):
                for j, site2 in enumerate(use_motifs2):
                    # Prepare data for each combination
                    class1_site1_df = pd.DataFrame({
                        'label': f'{site1}({use_class1_n})',
                        assay: self.class2_merged_data.loc[self.class1].query(f'{site1}==@use_class1_n')[assay].values
                    })
                    class1_site2_df = pd.DataFrame({
                        'label': f'{site2}({use_class1_n})',
                        assay: self.class2_merged_data.loc[self.class1].query(f'{site2}==@use_class1_n')[assay].values
                    })
                    class2_site12_df = pd.DataFrame({
                        'label': f'{site1}&\n{site2}({use_class2_n}:{use_class2_n})',
                        assay: self.class2_merged_data.loc[self.class2].query(
                            f'{site1}=={use_class2_n} & {site2}=={use_class2_n}')[assay].values
                    })
    
                    df = pd.concat([class1_site1_df, class1_site2_df, class2_site12_df]).dropna()
                    assay_data[(i, j)] = {
                        'data': df,
                        'site1': site1,
                        'site2': site2
                    }
            data_dict[assay] = assay_data
    
        self._create_class2_boxplots(data_dict)

    def _create_class2_boxplots(self, data_dict):
        """Create boxplots for class 2 interactions."""
        for assay, assay_data in data_dict.items():
            site1_labels = sorted(set(key[0] for key in assay_data))
            site2_labels = sorted(set(key[1] for key in assay_data))
            nrows = len(site1_labels)
            ncols = len(site2_labels)
    
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, 
                                   figsize=[2*ncols, 3*nrows], tight_layout=True)
    
            for (i, j), entry in assay_data.items():
                ax = axes[i, j]
                df = entry['data']
                site1 = entry['site1']
                site2 = entry['site2']
    
                sns.boxplot(data=df, x='label', y=assay, ax=ax, palette=sns.color_palette('colorblind'))
                sns.stripplot(data=df, x='label', y=assay, ax=ax, color='black', jitter=True)
    
                ax.xaxis.set_tick_params(rotation=45)
                ax.yaxis.set_tick_params(which='both', labelleft=True)
                ax.set_xlabel(None)
                ax.set_ylabel('log2(Activity)')
                ax.set_title(f'{site1}&{site2}')
    
                # Add significance annotation
                try:
                    adjp = self.class2_regression_results.query(
                        f'index in [@site1+"&"+@site2, @site2+"&"+@site1] & assay==@assay')['adjp'].values[0]
                    self._add_significance_annotation(ax, adjp, alpha=STRICT_FDR_ALPHA)
                except (IndexError, KeyError):
                    pass
    
                # Hide diagonal (same motif comparisons)
                if site1 == site2:
                    ax.clear()
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    ax.set_facecolor((1, 1, 1, 1))
    
                ax.set_ylim(*DEFAULT_PLOT_YLIM)
    
            print(assay)
            plt.show()
     
    def _merge_class2_data(self, motifs=None, features=None, assays=None):
        """Merge data for class 2 analysis with motif combinations."""
        if motifs is None:
            motifs = self.motifs
        if features is None:
            features = self.template + self.class1 + self.class2
        if assays is None:
            assays = self.assays
            
        logFC = self.logFC_reps.loc[features]
        nsite = self.pattern_table.loc[features]

        combination_labels = []

        # Create combination features
        for pair in itertools.combinations(motifs, 2):
            combination = np.sqrt(nsite.loc[self.class2, pair[0]] * nsite.loc[self.class2, pair[1]])
            nsite[f'{pair[0]}&{pair[1]}'] = combination
            combination_labels.append(f'{pair[0]}&{pair[1]}')
            
        nsite = nsite.fillna(0)

        self.combination_labels = combination_labels
        self.class2_merged_data = logFC.join(nsite)
                
    def _perform_class2_regression(self):
        """Perform regression analysis for class 2 motif interactions."""
        motifs = self.motifs + self.combination_labels
        results = []
        
        for assay in self.assays:
            y = self.class2_merged_data[assay].dropna()
            X = self.class2_merged_data[motifs][self.class2_merged_data[assay].notna()]
    
            est = sm.OLS(y, X).fit()
    
            df = pd.DataFrame({
                'p': est.pvalues.loc[self.combination_labels],
                'coef': est.params.loc[self.combination_labels]
            }).reset_index()
            df['assay'] = assay
            df['sig'], df['adjp'] = statsmodels.stats.multitest.fdrcorrection(
                df['p'], is_sorted=False, alpha=STRICT_FDR_ALPHA)

            results.append(df)
        
        return pd.concat(results)

    def plot_class2_interaction_network(self, use_motifs=None, use_assay='lentiMPRA'):
        """Plot network visualization of significant motif interactions."""
        if use_motifs is None:
            use_motifs = self.motifs

        # Parse interaction pairs
        self.class2_regression_results[['site1', 'site2']] = self.class2_regression_results['index'].str.split('&', expand=True)

        # Create matrices for visualization
        df_adjp = pd.DataFrame(np.nan, index=use_motifs, columns=use_motifs)
        df_adjp.update(self.class2_regression_results.query('assay==@use_assay').pivot_table(
            index='site1', columns='site2', values='adjp'))
        df_adjp = -np.log10(df_adjp)
        
        df_coef = pd.DataFrame(np.nan, index=use_motifs, columns=use_motifs)
        df_coef.update(self.class2_regression_results.query('assay==@use_assay').pivot_table(
            index='site1', columns='site2', values='coef'))

        # Build network graph
        G = nx.Graph()
        nodes = df_adjp.columns
        for node in nodes:
            G.add_node(node)

        # Add significant edges only
        for i, source in enumerate(df_adjp.index):
            for j, target in enumerate(df_adjp.columns):
                adjp_value = df_adjp.iloc[i, j]
                coef_value = df_coef.iloc[i, j]
                
                if not np.isnan(adjp_value) and adjp_value > -np.log10(STRICT_FDR_ALPHA):
                    if not np.isnan(coef_value):
                        G.add_edge(source, target, weight=adjp_value, coef=coef_value)

        self._draw_interaction_network(G)

    def _draw_interaction_network(self, G):
        """Draw the interaction network with proper styling."""
        pos = nx.circular_layout(G)

        plt.figure(figsize=NETWORK_PLOT_SIZE, facecolor='white')
        ax = plt.gca()
        ax.set_facecolor('white')

        # Draw nodes and labels
        nx.draw_networkx_nodes(G, pos, node_size=800, node_color='lightblue')
        nx.draw_networkx_labels(G, pos, font_size=12, font_color='black')
            
        # Prepare edge styling
        edges = G.edges(data=True)
        edge_weights = [d['weight'] for _, _, d in edges]  # -log10(adjp) values
        edge_signs = [1 if d['coef'] > 0 else 0 for _, _, d in edges]  # Signs of coefficients

        def scale_edge_width(weights):
            """Scale edge width based on significance."""
            return 4 * np.log2(weights)
                
        scaled_weights = scale_edge_width(edge_weights)

        # Color edges based on effect direction
        red_cmap = cm.get_cmap('Reds')
        blue_cmap = cm.get_cmap('Blues')
        max_weight = max(edge_weights) if edge_weights else 1
        norm = mcolors.Normalize(vmin=0, vmax=max_weight)

        edge_colors = []
        for weight, sign in zip(edge_weights, edge_signs):
            if sign == 1:  # Positive coefficient
                edge_colors.append(red_cmap(norm(weight)))
            else:  # Negative coefficient
                edge_colors.append(blue_cmap(norm(weight)))

        nx.draw_networkx_edges(G, pos, width=scaled_weights, edge_color=edge_colors)

        # Add legend
        legend_items = []
        for level in [2, 4, 8]:
            if level <= max_weight:
                legend_items.append(
                    Line2D([0], [0], color='gray', solid_capstyle='butt',
                          lw=scale_edge_width(level), label=f"-log10(adj-p) = {level:.1f}"))

        plt.legend(handles=legend_items, loc="upper left", fontsize=10, frameon=True)
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    def _perform_permutation_anova(self):
        """Perform ANOVA analysis for permutations."""
        results = []
        
        for assay in self.assays:
            assay_results = []
            for permutation in itertools.combinations(self.motifs, 4):
                permutation = list(permutation)
                permutation_ids = self.class3_permutation_data[permutation].all(axis=1)
    
                permutations = self.class3_permutation_data.loc[permutation_ids, assay]
            
                id_groups = permutations.groupby('id')
                f_stats, pvalue = stats.f_oneway(*[g.dropna().values for _, g in id_groups])

                # Calculate effect size as range of group means
                group_means = id_groups.mean()
                delta = group_means.max() - group_means.min()

                assay_results.append([assay, permutation, delta, pvalue])

            assay_df = pd.DataFrame(assay_results, columns=['assay', 'permutation', 'delta', 'pvalue'])
            sig, adjp = statsmodels.stats.multitest.fdrcorrection(
                assay_df['pvalue'], is_sorted=False, alpha=STRICT_FDR_ALPHA)
            assay_df['adjp'] = adjp
            assay_df['-log10(adjp)'] = -np.log10(adjp)

            results.append(assay_df)

        return pd.concat(results)

    def plot_permutation_anova_volcano(self):
        """Plot volcano plot for permutation ANOVA results."""
        plt.figure(figsize=PERMUTATION_PLOT_SIZE, tight_layout=True)
        
        sns.scatterplot(data=self.permutation_anova_results, x='delta', y='-log10(adjp)', 
                       hue='assay', palette=self.colors[:len(self.assays)])
        
        # Add significance threshold line
        plt.hlines(y=2, xmin=0.5, xmax=3, linestyles='dashed', colors='black')
        plt.annotate('adj-p = 0.01', xy=(3, 3), ha='right', va='center')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=16, markerscale=2)
        
        plt.ylabel('-log10(adj-p)')
        plt.xlabel('Best permutation log2(Activity)\n- Worst permutation log2(Activity)')

        plt.show()

    def plot_permutation_comparison(self, permutation, sortby):
        """Plot comparison of specific permutation across conditions."""
        colors = {tf: color for tf, color in zip(permutation, self.colors)}

        fig, axes = plt.subplots(ncols=len(self.assays)+1, nrows=1, tight_layout=True,
                               figsize=[3*len(self.assays), 9])

        # Filter data for the specified permutation
        permutation_ids = self.class3_permutation_data[permutation].all(axis=1)
        df = self.class3_permutation_data.loc[permutation_ids, self.assays]
        order = df[sortby].groupby('id').mean().sort_values(ascending=False).index

        # Draw permutation pattern visualization
        for i, elements in enumerate(order.str.split('_')):
            for j, element in enumerate(elements):
                rect = plt.Rectangle((j, i), 1, 1, color=colors[element])
                axes[0].add_patch(rect)
        
            axes[0].text(j+1, i+0.625, '-mP')

        axes[0].set_yticks([])
        axes[0].set_xticks([])
        axes[0].set_xlim(0, 4)
        axes[0].set_ylim(0, len(order))
        axes[0].invert_yaxis() 
        
        # Add legend
        handles = {key: plt.Circle(0, 0, color=color) for key, color in colors.items()}
        axes[0].legend(handles.values(), handles.keys(), loc='upper right', 
                      bbox_to_anchor=(1.15, 1.075), ncol=2, columnspacing=0.5)

        # Plot activity distributions for each assay
        for i, assay in enumerate(self.assays):
            ax = axes[i+1]
        
            # Get significance p-value
            adjp = self.permutation_anova_results[
                self.permutation_anova_results['permutation'].apply(
                    lambda x: Counter(x) == Counter(permutation))
            ].query('assay==@assay')['adjp']
            
            # Prepare data for plotting
            s = df[assay].groupby('id').apply(lambda x: list(x)).reindex(order)
            data = pd.DataFrame(dict(zip(order, s))).melt()
            
            # Create plots
            sns.stripplot(data=data, y='variable', x='value', ax=ax, color='black', orient='h')
            sns.boxplot(data=data, y='variable', x='value', ax=ax, orient='h')
            
            # Add significance annotation
            ax.text(0.0, -0.02, f'adj-p = {adjp.iloc[0]:.3f}', 
                   transform=ax.transAxes, fontsize=14)
            
            ax.set_xlabel(assay)
            ax.set_yticks([])
            ax.set_ylabel(None)
            ax.xaxis.set_label_position('top')
            ax.xaxis.tick_top()

        plt.show()

    def calculate_positional_enrichment(self, size=POSITION_ENRICHMENT_SIZE):
        """Calculate positional enrichment of motifs in permutations."""
        all_patterns = perm(len(self.motifs), 4)  # All possible permutations
        motif_at_position_k = perm(len(self.motifs)-1, 4-1)
        expected_frequency = (motif_at_position_k / all_patterns) * size

        results = []
        
        for assay in self.assays:
            # Sort permutations by activity
            order = self.class3_permutation_data[assay].groupby('id').mean().sort_values().dropna().reset_index()

            # Get top and bottom performers
            samples = {
                'bottom': order[:size]['id'].str.split('_', expand=True),
                'top': order[-size:]['id'].str.split('_', expand=True)
            }

            for label, data in samples.items():
                # Count motif occurrences at each position
                frequency = data.apply(pd.value_counts).fillna(0)
                for motif in self.motifs:
                    if motif not in frequency.index:
                        frequency.loc[motif] = 0
                frequency.sort_index(inplace=True)

                # Calculate enrichment (odds ratio)
                enrichment = frequency / expected_frequency
                enrichment = enrichment.reset_index().melt(id_vars='index')
                enrichment.columns = ['motif', 'position', 'odds']
                
                # Calculate p-values using hypergeometric test
                p_values = frequency.applymap(
                    lambda x: self._calculate_hypergeometric_pvalue(
                        x, HYPERGEO_TOTAL_POPULATION, HYPERGEO_SUCCESS_STATES, size)
                ).melt()
                
                # Multiple testing correction
                sig, adjp = statsmodels.stats.multitest.fdrcorrection(p_values['value'], is_sorted=False)

                enrichment['adjp'] = adjp
                enrichment['label'] = label
                enrichment['assay'] = assay

                results.append(enrichment)

        return pd.concat(results)

    def plot_positional_motif_enrichment(self, assays=None, size=POSITION_ENRICHMENT_SIZE):
        """Plot positional enrichment of motifs."""
        if assays is None:
            assays = self.assays
            
        enrichment_results = self.calculate_positional_enrichment(size=size)

        # Setup colors and axis order
        colors = {motif: color for motif, color in zip(self.motifs, self.colors)}
        ax_order = {motif: i for i, motif in enumerate(self.motifs)}

        for assay in assays:
            print(assay)
            
            for label in ['top', 'bottom']:
                df = enrichment_results.query('assay == @assay & label==@label')
                fig, axes = plt.subplots(ncols=len(self.motifs), nrows=1, tight_layout=True,
                                       figsize=[9, 4], sharex=True, sharey=True)
                fig.suptitle(f'{label}{size} permutations')
                
                axes[0].set_ylabel('odds ratio')

                for _, row in df.iterrows():
                    ax = axes[ax_order[row['motif']]]
                    ax.bar(row['position'], row['odds']-1, bottom=1, color=colors[row['motif']])

                    # Add significance annotations
                    q = row['adjp']
                    if q < 0.05:
                        if q < 1e-4:
                            annot = '****'
                        elif q < 1e-3:
                            annot = '***'
                        elif q < 1e-2:
                            annot = '**'
                        elif q < 5e-2:
                            annot = '*'

                        locus = row['position'] + 0.35
                        height = row['odds'] - 0.08 if row['odds'] - 1 < 0 else row['odds'] + 0.08
                            
                        ax.text(locus, height, annot, ha='center', va='center', rotation=90)
                    
                    ax.set_xticks(ticks=[0, 1, 2, 3], labels=[1, 2, 3, 4])
                    ax.set_xlabel(row['motif'])

                plt.show()

    def _calculate_hypergeometric_pvalue(self, k, N, K, n):
        """
        Calculate hypergeometric test p-value.
        
        Args:
            k: Number of observed successes
            N: Total population size
            K: Number of success states in population
            n: Number of draws
        """
        p_obs = hypergeom.pmf(k, N, K, n)
        p_value = np.sum(hypergeom.pmf(i, N, K, n) 
                         for i in range(0, n+1) 
                         if hypergeom.pmf(i, N, K, n) <= p_obs)
        return p_value