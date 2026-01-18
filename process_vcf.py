import sys
import os
import re
import gzip
import argparse

def parse_gt(gt_str):
    """
    Parse genotype string (e.g., '0/1', '1|0', './.')
    Returns a tuple of alleles (int) or None if missing/invalid.
    """
    if '.' in gt_str:
        return None
    
    # Split by / or |
    alleles = re.split(r'[/|]', gt_str)
    try:
        return [int(a) for a in alleles]
    except ValueError:
        return None


def process_vcf(vcf_path, prefix, filter_str, region, min_depth, exclude_chroms=None):
    if not os.path.exists(vcf_path):
        print(f"Error: File {vcf_path} not found.")
        return

    # Parse region if provided
    region_chrom = None
    region_start = None
    region_end = None
    if region:
        try:
            # Expect format chrom:start-end
            if ':' in region and '-' in region:
                r_chrom, r_range = region.split(':')
                r_start, r_end = r_range.split('-')
                region_chrom = r_chrom
                region_start = int(r_start)
                region_end = int(r_end)
            else:
                print("Error: Region format must be chrom:start-end")
                return
        except ValueError:
            print("Error: Region format must be chrom:start-end with integer coordinates")
            return

    # Determine how to open the file
    if vcf_path.endswith('.gz'):
        open_func = gzip.open
        mode = 'rt'
    else:
        open_func = open
        mode = 'r'

    output_handles = {}
    exp_samples = []
    
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
                        print("Error: VCF file must have at least one sample (Bulk).")
                        return
                    
                    # parts[9] is the first sample (Reference)
                    # parts[10:] are experiment samples
                    ref_sample_name = parts[9]
                    exp_sample_names = parts[9:]
                    
                    print(f"Bulk sample: {ref_sample_name}")
                    print(f"Experiment samples: {exp_sample_names[1:]}")
                    
                    # Create output files for experiment samples
                    for sample in exp_sample_names:
                        # Apply prefix
                        out_name = f"{prefix}{sample}.txt"
                        try:
                            fh = open(out_name, 'w')
                            # Header for output file
                            fh.write("#chrom\tpos\tdepth1\tdepth2\n")
                            output_handles[sample] = fh
                            exp_samples.append(sample)
                        except IOError as e:
                            print(f"Error creating output file for {sample}: {e}")
                            for fh in output_handles.values():
                                fh.close()
                            return
                    continue
                # print(exp_samples)
                # raise

                # Data lines
                parts = line.split('\t')
                chrom = parts[0]
                pos = parts[1]
                filter_val = parts[6]

                if exclude_chroms and chrom in exclude_chroms:
                    continue

                # only process SNVs
                ref = parts[3]
                alt = parts[4].split(',')
                if len(ref) != 1 or any(len(a) != 1 for a in alt):
                    continue

                # Filter by FILTER column
                if filter_str and filter_val != filter_str:
                    continue
                
                # Filter by Region
                if region_chrom:
                    if chrom != region_chrom:
                        continue
                    try:
                        pos_int = int(pos)
                        if not (region_start <= pos_int <= region_end):
                            continue
                    except ValueError:
                        continue

                format_str = parts[8]
                
                format_keys = format_str.split(':')
                
                try:
                    gt_idx = format_keys.index('GT')
                    ad_idx = format_keys.index('AD')
                except ValueError:
                    # GT or AD missing in FORMAT
                    continue

                # Process Reference Sample (Index 9)
                ref_data_str = parts[9]
                ref_data = ref_data_str.split(':')
                
                gt_str = ref_data[gt_idx]
                alleles = parse_gt(gt_str)
                
                if not alleles or len(alleles) != 2:
                    continue
                
                # Check if heterozygous
                if alleles[0] == alleles[1]:
                    continue

                # Check minimum depth if specified
                if min_depth is not None:
                    ad_str = ref_data[ad_idx]
                    if ad_str == '.':
                        continue
                    try:
                        ad_values = [int(x) for x in ad_str.split(',')]
                    except ValueError:
                        continue
                    total_depth = ad_values[alleles[0]] + ad_values[alleles[1]]
                    if total_depth < min_depth:
                        continue
                
                # It is heterozygous. Alleles are alleles[0] and alleles[1]
                a1, a2 = alleles[0], alleles[1]
                
                # Process Experiment Samples
                for i, sample_name in enumerate(exp_samples):
                    exp_idx = 9 + i
                    if exp_idx >= len(parts):
                        break
                        
                    exp_data_str = parts[exp_idx]
                    
                    # Handle missing data './.' or just '.'
                    if exp_data_str == '.' or exp_data_str == './.':
                        continue

                    exp_data = exp_data_str.split(':')
                    ad_str = exp_data[ad_idx]
                    if ad_str == '.':
                        continue
                        
                    try:
                        ad_values = [int(x) for x in ad_str.split(',')]
                    except ValueError:
                        continue
                    
                    # We need depths for allele a1 and a2
                    d1 = ad_values[a1]
                    d2 = ad_values[a2]

                    # Check minimum depth if specified
                    if min_depth is not None:
                        total_depth = d1 + d2
                        if total_depth < min_depth:
                            continue
                        
                    output_handles[sample_name].write(f"{chrom}\t{pos}\t{d1}\t{d2}\n")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Close all handles
        for fh in output_handles.values():
            fh.close()

if __name__ == "__main__":
    class CustomFormatter(argparse.HelpFormatter):
        def _format_action_invocation(self, action):
            if not action.option_strings or action.nargs == 0:
                return super()._format_action_invocation(action)
            default = self._get_default_metavar_for_optional(action)
            args_string = self._format_args(action, default)
            return ', '.join(action.option_strings[:-1]) + f', {action.option_strings[-1]} {args_string}'
        
    parser = argparse.ArgumentParser(description="Process VCF file to extract allele depths for bulk heterozygous sites.\nFirst sample in VCF is considered the bulk sample.",
                                     formatter_class=CustomFormatter)
    parser.add_argument("input_vcf", help="Path to the input VCF file")
    parser.add_argument("-p", "--prefix", default="", help="Prefix for output files")
    parser.add_argument("-f", "--filter", dest="filter_str", help="Filter string (exact match in FILTER column)")
    parser.add_argument("-R", "--region", help="Region splice (format: chrom:start-end)")
    parser.add_argument("-e", "--exclude_chroms", nargs='*', help="List of chromosomes to exclude from analysis")
    parser.add_argument("-d", "--depth", type=int, help="Minimum depth")
    parser.add_argument("-v", "--version", action="version", version="process_vcf.py 0.1.0")
    
    args = parser.parse_args()
    
    process_vcf(args.input_vcf, args.prefix, args.filter_str, args.region, args.depth, args.exclude_chroms)
