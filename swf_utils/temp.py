import sys
import os
import pandas as pd

# Make sure we can import our categorizer module
script_dir = os.path.abspath(os.path.join(os.getcwd(), 'swf_utils'))
sys.path.insert(0, script_dir)

from swf_categorizer import process_swf

# Adjust the SWF path as needed
swf_path = '/home/poornav/cloudsim-simulator/SDSC-SP2-1998-4.2-cln.swf'
anomalies_df, labeled_df, summary_df = process_swf(swf_path, anomaly_pct=1.0)

# Prepare Excel writer
output_path = '/home/poornav/cloudsim-simulator/swf_categories_with_stats.xlsx'
with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
    # 1) Anomalies sheets
    anomalies_df.to_excel(writer, sheet_name='Anomalies', index=False)  
    anomalies_df.describe().transpose().to_excel(writer, sheet_name='Anomalies Stats')
    
    # 2) Per-category sheets
    for cat in labeled_df['Category'].unique():
        # Limit sheet name to 31 characters
        sheet_name = cat[:31]
        stats_name = f'{sheet_name} Stats'[:31]
        
        subset = labeled_df[labeled_df['Category'] == cat]
        subset.to_excel(writer, sheet_name=sheet_name, index=False)
        subset.describe().transpose().to_excel(writer, sheet_name=stats_name)
    
    # 3) Summary sheet
    summary_df.to_excel(writer, sheet_name='Summary', index=False)

print(f"Generated Excel workbook: {output_path}")
