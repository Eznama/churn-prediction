import os, sys
required_dirs = ['data/raw','data/processed','notebooks','src','models','reports/figures']
required_files = ['README.md','.gitignore','requirements.txt','data/raw/telco_churn.csv']
missing = [p for p in required_dirs+required_files if not os.path.exists(p)]
print('Missing:', missing if missing else 'None')
print('Python exe:', sys.executable)
