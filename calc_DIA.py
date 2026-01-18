import numpy as np
import pandas as pd
import scipy.stats as sts
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os, argparse, natsort, sys

def calc_amplicon(rd, ad, p = 0.5, bootstrap = 0):
    if len(rd) != len(ad):
        raise ValueError("Length of RD and AD must be the same.")
    depth = rd + ad
    mask = (depth >= 10)
    depth = depth[mask]
    
    if len(depth) == 0:
        if bootstrap == 0:
            return float('nan')
        else:
            return float('nan'), []

    freq = ad[mask] / depth
    numerator = (depth - 1)
    
    # Safety check for p
    if type(p) == float:
        if p <= 0 or p >= 1:
            if bootstrap == 0:
                return float('inf')
            else:
                return float('inf'), []
    elif type(p) == np.ndarray:
        p = p[mask]
        valid_p_mask = (p > 0) & (p < 1)
        if np.sum(valid_p_mask) == 0:
            if bootstrap == 0:
                return float('inf')
            else:
                return float('inf'), []
        depth = depth[valid_p_mask]
        freq = freq[valid_p_mask]
        numerator = numerator[valid_p_mask]
        p = p[valid_p_mask]

    with np.errstate(divide='ignore', invalid='ignore'):
        denominator = (depth * ((freq - p) ** 2) / (p * (1-p)) - 1)
    if denominator.sum() <= 1e-8:
        rlt = float('inf')
    else:
        rlt = numerator.sum() / denominator.sum()

    if bootstrap == 0:
        return rlt
    bootstrap_rlt = []
    index = np.arange(len(depth))
    for _ in range(bootstrap):
        index_bs = np.random.choice(index, size = len(index), replace = True)
        numerator_bs = numerator[index_bs]
        denominator_bs = denominator[index_bs]
        if denominator_bs.sum() <= 1e-8:
            rlt_bs = float('inf')
        else:
            rlt_bs = numerator_bs.sum() / denominator_bs.sum()
        bootstrap_rlt.append(rlt_bs)
    return rlt, bootstrap_rlt


def process_region(chrom, ploidy, records):
    if not records:
        return None

    # records is list of (pos, RD, AD)
    data = np.array(records)
    # data[:, 0] is pos, data[:, 1] is RD, data[:, 2] is AD

    sort_indices = np.argsort(data[:, 0])
    data = data[sort_indices]
    
    RD = data[:, 1]
    AD = data[:, 2]
    
    # Infer p from data
    depth = RD + AD
    mask = depth >= 10
    if np.sum(mask) == 0:
        # Not enough data
        if args.verbosity != 'silent':
            print(f"Warning: Region {chrom}:{data[0,0]}-{data[-1,0]} (ploidy {ploidy}): Not enough data (depth>=10)")
            print("Skipping DIA estimation for this region.\n")
        return {'chrom': chrom, 'ploidy': ploidy, 'p': 0.5, 'data': data}

    valid_RD = RD[mask].sum()
    valid_depth = depth[mask].sum()
    freqs = valid_RD / valid_depth
    p_estimated = round(freqs * ploidy) / ploidy
    # Safety check for p_estimated
    if p_estimated <= 0 or p_estimated >= 1:
        # Out of bounds
        if args.verbosity != 'silent':
            print(f"Warning: Region {chrom}:{data[0,0]}-{data[-1,0]} (ploidy {ploidy}): Estimated p={p_estimated} out of bounds")
            print("Skipping DIA estimation for this region.\n")
        p_estimated = 0.0 if p_estimated <= 0 else 1.0
        return {'chrom': chrom, 'ploidy': ploidy, 'p': p_estimated, 'data': data}
    
    # if args.verbosity != 'silent':
    #     test = sts.binomtest(valid_RD, valid_depth, p_estimated).pvalue
    #     if test < 0.01:
    #         # Warning: estimated p fails binomial test
    #         print(f"Warning: Region {chrom}:{data[0,0]}-{data[-1,0]} (ploidy {ploidy}):\nEstimated p={p_estimated} fails binomial test (p={test})")
    #         print(f"Total RD={valid_RD}, Total Depth={valid_depth}")
    #         print("Consider check data quality or ploidy assumption.\n")
    
    return {'chrom': chrom, 'ploidy': ploidy, 'p': p_estimated, 'data': data}


def split_region(chrom, ploidy, records, prev_pos, region_size = 1_000_000, min_records_per_chunk = 100):
    res = process_region(chrom, ploidy, records)
    if res is None:
        return []
    
    temp_pos = res['data'][:,0]
    
    # actual_region_size = temp_pos[-1] - prev_pos + 1
    # if actual_region_size > region_size:
    #     actual_region_size = int(np.ceil(actual_region_size / np.ceil(actual_region_size / region_size)))
    prev_ind = 0
    next_pos = prev_pos + region_size - 1
    cutoff_ind = np.searchsorted(temp_pos, next_pos, side='right')
    chunks = []
    
    while True:
        if cutoff_ind == len(temp_pos):
            next_pos = temp_pos[-1]
        # if cutoff_ind - prev_ind < min_records_per_chunk:
            # if args.verbosity != 'silent':
            #     print(f"Warning: Region {chrom}:{prev_pos}-{next_pos} has only {cutoff_ind - prev_ind} loci.")
            #     print("Region DIA estimation may be inaccurate due to insufficient data.\n")
        chunk_data = res['data'][prev_ind:cutoff_ind]
        chunks.append({
            'chrom': res['chrom'],
            'ploidy': res['ploidy'],
            'p': res['p'],
            'region_info': f"{res['chrom']}:{prev_pos}-{next_pos}",
            'data': chunk_data
        })
        if cutoff_ind >= len(temp_pos):
            break
        prev_ind = cutoff_ind
        prev_pos = next_pos + 1
        next_pos = prev_pos + region_size - 1
        cutoff_ind = np.searchsorted(temp_pos, next_pos, side='right')
    return chunks


def load_data(filepath, region_size = 1_000_000, min_records_per_chunk = 100):
    current_chrom = None
    current_ploidy = None
    region_records = [] 
    all_chunks = []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            if len(parts) < 4:
                continue
                
            chrom = parts[0]
            try:
                pos = int(parts[1])
                rd = int(parts[2])
                ad = int(parts[3])
                if len(parts) >= 5:
                    ploidy = int(parts[4])
                else:
                    ploidy = 2
            except ValueError:
                if args.verbosity != 'silent':
                    print(f"Warning: Skipping invalid line: {line}\n")
                continue

            # Check if region changed
            if chrom != current_chrom or ploidy != current_ploidy:
                # Process previous region
                if current_chrom is not None:
                    if len(all_chunks) == 0 or all_chunks[-1]['chrom'] != current_chrom:
                        prev_pos = 1
                    else:
                        prev_pos = all_chunks[-1]['data'][-1, 0] + 1
                    res = split_region(current_chrom, current_ploidy, region_records, prev_pos, region_size, min_records_per_chunk)
                    all_chunks.extend(res)
                
                # Start new region
                current_chrom = chrom
                current_ploidy = ploidy
                region_records = []

            region_records.append((pos, rd, ad))

        # Process last region
        if current_chrom is not None:
            if len(all_chunks) == 0 or all_chunks[-1]['chrom'] != current_chrom:
                prev_pos = 1
            else:
                prev_pos = all_chunks[-1]['data'][-1, 0] + 1
            res = split_region(current_chrom, current_ploidy, region_records, prev_pos, region_size, min_records_per_chunk)
            all_chunks.extend(res)
    
    if not all_chunks:
        return pd.DataFrame(columns=['chrom', 'pos', 'rd', 'ad', 'p', 'ploidy', 'region_info'])

    chunk_lengths = [len(c['data']) for c in all_chunks]
    
    chroms = np.repeat([c['chrom'] for c in all_chunks], chunk_lengths)
    region_infos = np.repeat([c['region_info'] for c in all_chunks], chunk_lengths)
    ploidies = np.repeat([c['ploidy'] for c in all_chunks], chunk_lengths)
    ps = np.repeat([c['p'] for c in all_chunks], chunk_lengths)
    
    all_data = np.concatenate([c['data'] for c in all_chunks], axis=0)
    
    df = pd.DataFrame({
        'chrom': chroms,
        'pos': all_data[:, 0],
        'rd': all_data[:, 1],
        'ad': all_data[:, 2],
        'p': ps,
        'ploidy': ploidies,
        'region_info': region_infos
    })
    
    return df


def calculate_region_stats(df, bootstrap_n=500):
    results = []
    # Group by region_info. 
    # region_info is like "chrom:start-end"
    
    if df.empty:
        return pd.DataFrame(columns=['chrom', 'start', 'end', 'ploidy', 'p_est', 'dia', 'dia_upper', 'dia_lower'])

    grouped = df.groupby('region_info')
    
    for region_info, group in grouped:
        rd = group['rd'].values
        ad = group['ad'].values
        chrom = group.iloc[0]['chrom']
        ploidy = group.iloc[0]['ploidy']
        p = group.iloc[0]['p']
        
        try:
            # region_info: chrom:start-end
            _, range_part = region_info.split(':')
            start_str, end_str = range_part.split('-')
            start = int(start_str)
            end = int(end_str)
        except:
            start = group['pos'].min()
            end = group['pos'].max()
            
        dia, bs_rlt = calc_amplicon(rd, ad, p=p, bootstrap=bootstrap_n)
        
        if bootstrap_n > 0 and isinstance(bs_rlt, list) and len(bs_rlt) > 0:
            with np.errstate(divide='ignore', invalid='ignore'):
                dia_lower = np.percentile(bs_rlt, 2.5)
                dia_upper = np.percentile(bs_rlt, 97.5)
        else:
            dia_lower = np.nan
            dia_upper = np.nan
            
        results.append({
            'chrom': chrom,
            'start': start,
            'end': end,
            'ploidy': ploidy,
            'p_est': p,
            'dia': dia,
            'dia_upper': dia_upper,
            'dia_lower': dia_lower,
            'region_info': region_info
        })
        
    df = pd.DataFrame(results)

    # Sort using natsort for chroms to ensure correct order
    chroms = df['chrom'].unique()
    sorted_chroms = natsort.natsorted(chroms)
    df['chrom'] = pd.Categorical(df['chrom'], categories=sorted_chroms, ordered=True)
    df = df.sort_values(by=['chrom', 'start']).reset_index(drop=True)

    # Interpolate inf values for plotting
    for col in ['dia', 'dia_lower', 'dia_upper']:
        plot_col = f'{col}_plot'
        df[plot_col] = df[col].replace([np.inf, -np.inf], np.nan)
        # Group by chrom to avoid interpolating across chromosomes
        with np.errstate(invalid='ignore'):
            df[plot_col] = df.groupby('chrom', observed=True)[plot_col].transform(
                lambda x: x.interpolate(method='linear', limit_direction='both')
            )
    
    df.replace(np.nan, np.inf, inplace=True)

    chrom_sizes = df.groupby('chrom', observed=True)['end'].max()
    sorted_chroms = natsort.natsorted(chrom_sizes.index)

    offsets = {}
    current_offset = 0
    gap = int(chrom_sizes.max() * 0.05)  # 5% of max chrom size as gap
    x_ticks_vals = []
    x_ticks_text = []

    for chrom in sorted_chroms:
        size = chrom_sizes[chrom]
        offsets[chrom] = current_offset
        x_ticks_vals.append(current_offset + size / 2)
        x_ticks_text.append(chrom)
        current_offset += (size + gap)
    
    df['plot_x'] = df.apply(lambda row: (row['start'] + row['end']) / 2 + offsets[row['chrom']], axis=1)

    # Handle single-row chromosomes
    chrom_counts = df['chrom'].value_counts()
    single_row_chroms = chrom_counts[chrom_counts == 1].index
    
    new_rows = []
    for chrom in single_row_chroms:
        mask = df['chrom'] == chrom

        idx = df[mask].index[0]
        row = df.loc[idx]
        
        # Update original row plot_x to start
        df.at[idx, 'plot_x'] = row['start'] + offsets[chrom]
        
        # Create new row with plot_x at end
        new_row = row.copy()
        new_row['plot_x'] = row['end'] + offsets[chrom]
        new_rows.append(new_row)
            
    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

    df.loc[:, 'hover_text_DIA'] = [f"{df.loc[i, 'region_info']}<br>DIA: {df.loc[i, 'dia']:.2f} (95% CI: {df.loc[i, 'dia_lower']:.2f}-{df.loc[i, 'dia_upper']:.2f})" for i in range(df.shape[0])]
    df.loc[:, 'hover_text_AF'] = [f"{df.loc[i, 'region_info']}<br>Estimated Allele Frequency: {df.loc[i, 'p_est']:.4f}" for i in range(df.shape[0])]
    df.loc[:, 'hover_text_CN'] = [f"{df.loc[i, 'region_info']}<br>Ploidy: {df.loc[i, 'ploidy']}" for i in range(df.shape[0])]
    return df, x_ticks_vals, x_ticks_text


def make_plot(wg_info, region_stats, x_vals, x_text, output_file):
    fig = make_subplots(
        rows=3, cols=2,
        specs=[[{"colspan": 2}, None],
            [{"secondary_y": True, "colspan": 2}, None],
            [{"colspan": 2}, None]],
        subplot_titles=(f"Whole Genome DIA Bootstrap Distribution<br>{wg_info['hover_text']}", 
                        "Variant Allele Frequency & Copy Number", 
                        "Split Region DIA"),
        vertical_spacing=0.1
    )

    # Fig1: Whole Genome eDIA Bootstrap Distribution
    fig.add_trace(
        go.Scatter(x=wg_info['kde_x'],
                   y=wg_info['kde_y'],
                   mode='lines',
                   name='DIA KDE',
                   hoverinfo='skip'),
        row=1, col=1
    )
    fig.add_vline(x=wg_info['dia'],
                  line_width = 3,
                  line_dash='dash',
                  line_color='red',
                  annotation_text='DIA',
                  row=1, col=1)
    
    # fig2 & 3: genomic tracks
    colors_cn = 'blue'
    colors_af = 'firebrick'
    colors_dia = 'teal'

    show_legend_group = True
    for chrom in region_stats['chrom'].unique():
        df_chrom = region_stats[region_stats['chrom'] == chrom]
        df_chrom = df_chrom.sort_values(by='plot_x')

        # Fig2: Ploidy & Allele Frequency
        fig.add_trace(
            go.Scatter(
                x=df_chrom['plot_x'],
                y=df_chrom['ploidy'],
                mode='lines',
                name='Ploidy',
                line=dict(color=colors_cn, dash='dot'),
                text=df_chrom['hover_text_CN'],
                hoverinfo='text',
                showlegend=show_legend_group
            ),
            row=2, col=1, secondary_y=True
        )

        fig.add_trace(
            go.Scatter(
                x=df_chrom['plot_x'],
                y=df_chrom['p_est'],
                mode='lines',
                name='Estimated Allele Frequency',
                marker=dict(color=colors_af),
                text=df_chrom['hover_text_AF'],
                hoverinfo='text',
                showlegend=show_legend_group
            ),
            row=2, col=1, secondary_y=False
        )
    
        # Fig3: Split Region DIA Estimate
        # CI Band
        fig.add_trace(
            go.Scatter(
                x=pd.concat([df_chrom['plot_x'], df_chrom['plot_x'][::-1]]),
                y=pd.concat([df_chrom['dia_upper'], df_chrom['dia_lower'][::-1]]),
                fill='toself',
                fillcolor='rgba(0,128,128,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo='skip',
                showlegend=show_legend_group,
                name='DIA 95% CI'
            ),
            row=3, col=1
        )

        # eDIA line
        fig.add_trace(
            go.Scatter(
                x=df_chrom['plot_x'],
                y=df_chrom['dia'],
                mode='lines',
                name='Region DIA',
                marker=dict(color=colors_dia),
                text=df_chrom['hover_text_DIA'],
                hoverinfo='text',
                showlegend=show_legend_group
            ),
            row=3, col=1
        )

        # Only show legend for first chromosome loop
        show_legend_group = False
    
    sample_name = os.path.basename(output_file).replace('.html', '')
    fig.update_layout(height=900, width=1200, title_text=f"{sample_name} DIA Analysis")

    # Set x-axis ticks to chromosome names
    for row in [2, 3]:
        fig.update_xaxes(
            row=row, col=1,
            tickvals = x_vals,
            ticktext = x_text,
            # title_text = "Genomic Position"
        )
    # Set y-axis titles
    fig.update_yaxes(title_text='Frequency', row=1, col=1)
    fig.update_yaxes(title_text='VAF', row=2, col=1, secondary_y=False, range=[0,1])
    fig.update_yaxes(title_text="Ploidy", row=2, col=1, secondary_y=True)
    fig.update_yaxes(title_text='DIA', row=3, col=1, type='log')

    fig.write_html(output_file)



if __name__ == "__main__":
    class CustomFormatter(argparse.HelpFormatter):
        def _format_action_invocation(self, action):
            if not action.option_strings or action.nargs == 0:
                return super()._format_action_invocation(action)
            default = self._get_default_metavar_for_optional(action)
            args_string = self._format_args(action, default)
            return ', '.join(action.option_strings[:-1]) + f', {action.option_strings[-1]} {args_string}'
        
    parser = argparse.ArgumentParser(description="Estimate DIA of whole genome and split regions.",
                                     formatter_class=CustomFormatter)
    parser.add_argument("input_file", help="Input file path")
    parser.add_argument("output_file", nargs="?", help="Output file path")
    parser.add_argument('-r', "--region_size", type=int, default=10_000_000, help="Region size to be splitted (default: 10,000,000)")
    parser.add_argument('-b', "--bootstrap", type=int, default=1000, help="Bootstrap number for whole genome (default: 1000)")
    parser.add_argument('-br', "--bootstrap_region", type=int, default=None, help="Bootstrap number for region (default: 0.5x of whole genome)")
    parser.add_argument('-q', "--quiet", action='store_const', const='quiet', dest='verbosity', help="Quiet mode, suppress output messages. Warnings will still be shown")
    parser.add_argument('-s', "--silent", action='store_const', const='silent', dest='verbosity', help="Silent mode, suppress output messages. Warnings will NOT be shown")
    parser.add_argument('-v', "--version", action="version", version="calc_DIA.py 0.1.0")
    parser.set_defaults(verbosity='normal')

    args = parser.parse_args()
    
    if args.bootstrap_region is None:
        args.bootstrap_region = int(args.bootstrap * 0.5)
        
    if not args.output_file:
        base, ext = os.path.splitext(args.input_file)
        if ext.lower() == '.txt':
            args.output_file = base + '.html'
        else:
            args.output_file = args.input_file + '.html'

    if os.path.exists(args.output_file):
        if not os.access(args.output_file, os.W_OK):
            print(f"Error: Cannot write to output file {args.output_file}. Check permissions.")
            sys.exit(1)
    else:
        parent_dir = os.path.dirname(args.output_file) or '.'
        if not os.path.exists(parent_dir):
            print(f"Error: Directory {parent_dir} does not exist.")
            sys.exit(1)
        if not os.access(parent_dir, os.W_OK):
            print(f"Error: Cannot write to directory {parent_dir}. Check permissions.")
            sys.exit(1)

    if args.verbosity == 'normal':
        print(f"Processing {args.input_file}...")
        print(f"Region size to be splitted: {args.region_size}")
        print(f"Bootstrap: {args.bootstrap}")
        print(f"Bootstrap Region: {args.bootstrap_region}")
        print(f"Output file: {args.output_file}\n")
    
    loaded = load_data(args.input_file, region_size=args.region_size)
    
    if args.verbosity == 'normal':
        print("Calculating Whole Genome DIA...")
    eDIA, bootstrap_eDIA = calc_amplicon(
        loaded['rd'].values,
        loaded['ad'].values,
        p=loaded['p'].values,
        bootstrap=args.bootstrap)
    
    if args.bootstrap > 0 and isinstance(bootstrap_eDIA, list) and len(bootstrap_eDIA) > 0:
        wg_lower = np.percentile(bootstrap_eDIA, 2.5)
        wg_upper = np.percentile(bootstrap_eDIA, 97.5)
        if args.verbosity == 'normal':
            print(f"Whole Genome DIA: {eDIA:.2f} (95% CI: {wg_lower:.2f}-{wg_upper:.2f})\n")
        temp_bootstrap_eDIA = [val for val in bootstrap_eDIA if val != float('inf')]
        if len(temp_bootstrap_eDIA) == 0:
            x = []
            y = []
        else:
            x = np.arange(min(temp_bootstrap_eDIA), max(temp_bootstrap_eDIA), (max(temp_bootstrap_eDIA)-min(temp_bootstrap_eDIA))/100)
            kde = sts.gaussian_kde(temp_bootstrap_eDIA)
            y = kde(x)
    else:
        if args.verbosity == 'normal':
            print(f"Whole Genome DIA: {eDIA:.2f}\n")
    wg_info = {
        'dia': eDIA,
        'boots': bootstrap_eDIA,
        'hover_text': f"DIA: {eDIA:.2f} (95% CI: {wg_lower:.2f}-{wg_upper:.2f})" if args.bootstrap > 0 else f"DIA: {eDIA:.2f}",
        'kde_x': x if args.bootstrap > 0 else [],
        'kde_y': y if args.bootstrap > 0 else [],
    }

    if args.verbosity == 'normal':
        print("Calculating Region Stats...")
    region_stats, x_vals, x_text = calculate_region_stats(loaded, bootstrap_n=args.bootstrap_region)
    
    if args.verbosity == 'normal':
        print("Generating Plot...")
    make_plot(wg_info, region_stats, x_vals, x_text, args.output_file)

    if args.verbosity == 'normal':
        print("Done.")
