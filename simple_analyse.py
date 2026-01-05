import sys
import os
import re
import gzip
import argparse
import numpy as np

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


def parse_gt(gt_str):
    """
    Parse genotype string (e.g., '0/1', '1|0', './.')
    Returns a list of alleles (int) or None if missing/invalid.
    """
    if '.' in gt_str:
        return None
    
    # Split by / or |
    alleles = re.split(r'[/|]', gt_str)
    try:
        return [int(a) for a in alleles]
    except ValueError:
        return None


def process_and_analyze(vcf_path, output_path, bootstrap_n, exclude_chroms=None, filter_str=None):
    if not os.path.exists(vcf_path):
        print(f"Error: File {vcf_path} not found.")
        return

    # Determine how to open the file
    if vcf_path.endswith('.gz'):
        open_func = gzip.open
        mode = 'rt'
    else:
        open_func = open
        mode = 'r'

    samples_data = {} # {sample_name: {'d1': [], 'd2': []}}
    exp_sample_names = []
    
    print(f"Reading VCF: {vcf_path}...")
    try:
        with open_func(vcf_path, mode) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                if line.startswith('##'):
                    continue
                
                if line.startswith('#CHROM'):
                    parts = line.split('\t')
                    if len(parts) < 10:
                        print("Error: VCF file must have at least one sample (Reference).")
                        return
                    
                    # parts[9] is the first sample (Reference)
                    # parts[10:] are experiment samples
                    ref_sample_name = parts[9]
                    exp_sample_names = parts[9:]
                    
                    print(f"Reference sample: {ref_sample_name}")
                    print(f"Experiment samples: {len(exp_sample_names)-1} samples found.")
                    
                    for sample in exp_sample_names:
                        samples_data[sample] = {'d1': [], 'd2': [], 'chrom': [], 'pos': []}
                    continue

                # Data lines
                parts = line.split('\t')

                chrom = parts[0]
                if exclude_chroms and chrom in exclude_chroms:
                    continue
                if filter_str and parts[6] != filter_str:
                    continue

                # only process SNVs
                ref = parts[3]
                alt = parts[4].split(',')
                if len(ref) != 1 or any(len(a) != 1 for a in alt):
                    continue

                format_str = parts[8]
                format_keys = format_str.split(':')
                
                try:
                    gt_idx = format_keys.index('GT')
                    ad_idx = format_keys.index('AD')
                except ValueError:
                    continue

                # Process Reference Sample (Index 9)
                ref_data_str = parts[9]
                ref_data = ref_data_str.split(':')
                
                if len(ref_data) <= gt_idx: 
                    continue
                
                gt_str = ref_data[gt_idx]
                alleles = parse_gt(gt_str)
                
                if not alleles or len(alleles) != 2:
                    continue
                
                # Check if heterozygous
                if alleles[0] == alleles[1]:
                    continue
                
                a1, a2 = alleles[0], alleles[1]

                ad_str = ref_data[ad_idx]
                if ad_str == '.':
                    continue
                ad_values = [int(x) for x in ad_str.split(',')]
                if ad_values[a1] + ad_values[a2] < 10:
                    continue

                
                # Process Experiment Samples
                for i, sample_name in enumerate(exp_sample_names):
                    exp_idx = 9 + i
                    if exp_idx >= len(parts):
                        break
                        
                    exp_data_str = parts[exp_idx]
                    if exp_data_str == '.' or exp_data_str == './.':
                        continue

                    exp_data = exp_data_str.split(':')
                    if len(exp_data) <= ad_idx:
                        continue
                        
                    ad_str = exp_data[ad_idx]
                    if ad_str == '.':
                        continue
                        
                    try:
                        ad_values = [int(x) for x in ad_str.split(',')]
                    except ValueError:
                        continue
                    
                    if a1 < len(ad_values) and a2 < len(ad_values):
                        d1 = ad_values[a1]
                        d2 = ad_values[a2]
                        if d1 + d2 < 10:
                            continue
                        
                        samples_data[sample_name]['d1'].append(d1)
                        samples_data[sample_name]['d2'].append(d2)
                        samples_data[sample_name]['chrom'].append(chrom)
                        samples_data[sample_name]['pos'].append(int(parts[1]))

    except Exception as e:
        print(f"An error occurred while reading VCF: {e}")
        return
        

    print("Analyzing samples...")
    try:
        with open(output_path, 'w') as out:
            out.write("sample_name\teDIA\tbootstrap_lower\tbootstrap_upper\n")
            
            for sample in exp_sample_names:
                data = samples_data[sample]
                if not data['d1']:
                    print(f"Warning: No data for sample {sample}")
                    out.write(f"{sample}\tNaN\tNaN\tNaN\n")
                    continue
                
                rd = np.array(data['d1'])
                ad = np.array(data['d2'])
                
                # Calculate eDIA
                # Using p=0.5 as default for fast analysis without region splitting
                dia, boots = calc_amplicon(rd, ad, p=0.5, bootstrap=bootstrap_n)
                
                lower = np.nan
                upper = np.nan
                
                if bootstrap_n > 0 and isinstance(boots, list) and len(boots) > 0:
                    with np.errstate(invalid='ignore'):
                        lower = np.percentile(boots, 2.5)
                        upper = np.percentile(boots, 97.5)
                    lower = lower if not np.isnan(lower) else np.inf
                    upper = upper if not np.isnan(upper) else np.inf
                
                out.write(f"{sample}\t{dia:.2f}\t{lower:.2f}\t{upper:.2f}\n")
                print(f"Processed {sample}: eDIA={dia:.2f} (95% CI: {lower:.2f}-{upper:.2f})")

    except IOError as e:
        print(f"Error writing output: {e}")

if __name__ == "__main__":
    class CustomFormatter(argparse.HelpFormatter):
        def _format_action_invocation(self, action):
            if not action.option_strings or action.nargs == 0:
                return super()._format_action_invocation(action)
            default = self._get_default_metavar_for_optional(action)
            args_string = self._format_args(action, default)
            return ', '.join(action.option_strings[:-1]) + f', {action.option_strings[-1]} {args_string}'
        
    parser = argparse.ArgumentParser(description="Fast eDIA analysis from VCF for all experiment samples.\nFirst sample in VCF is considered the bulk sample.",
                                     formatter_class=CustomFormatter)
    parser.add_argument("input_vcf", help="Path to input VCF file")
    parser.add_argument("output_tsv", nargs="?", help="Path to output TSV file")
    parser.add_argument("-f", "--filter", dest="filter_str", help="Filter string (exact match in FILTER column)")
    parser.add_argument("-b", "--bootstrap", type=int, default=1000, help="Number of bootstraps (default: 1000)")
    parser.add_argument("-e", "--exclude_chroms", nargs='*', help="List of chromosomes to exclude from analysis")
    parser.add_argument("-v", "--version", action="version", version="fast_analyse.py 0.1.0")

    args = parser.parse_args()

    if not args.output_tsv:
        base, ext = os.path.splitext(args.input_vcf)
        if ext == '.gz':
            base, ext = os.path.splitext(base)
        args.output_tsv = f"{base}_samples_eDIA.tsv"
    
    if os.path.exists(args.output_tsv):
        if not os.access(args.output_tsv, os.W_OK):
            print(f"Error: Cannot write to output file {args.output_tsv}. Check permissions.")
            sys.exit(1)
    else:
        parent_dir = os.path.dirname(args.output_tsv) or '.'
        if not os.path.exists(parent_dir):
            print(f"Error: Directory {parent_dir} does not exist.")
            sys.exit(1)
        if not os.access(parent_dir, os.W_OK):
            print(f"Error: Cannot write to directory {parent_dir}. Check permissions.")
            sys.exit(1)

    process_and_analyze(args.input_vcf, args.output_tsv, args.bootstrap, args.exclude_chroms, args.filter_str)