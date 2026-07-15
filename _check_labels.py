import pandas as pd

cols = ['duration','protocol_type','service','flag','src_bytes','dst_bytes',
'land','wrong_fragment','urgent','hot','num_failed_logins','logged_in',
'num_compromised','root_shell','su_attempted','num_root','num_file_creations',
'num_shells','num_access_files','num_outbound_cmds','is_host_login',
'is_guest_login','count','srv_count','serror_rate','srv_serror_rate',
'rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate',
'srv_diff_host_rate','dst_host_count','dst_host_srv_count',
'dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate',
'dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate',
'dst_host_rerror_rate','dst_host_srv_rerror_rate','label','difficulty']

for f in ['KDDTrain+.txt','KDDTrain+_20Percent.txt','train_test']:
    df = pd.read_csv('Train/'+f, header=None, names=cols)
    u = sorted(df['label'].unique().tolist())
    print('===', f, 'rows=', len(df), 'unique_labels=', len(u), '===')
    print(u)

# also check the saved class list file
import os
cf = 'Train/encoder_multiclass_23_classes.txt'
if os.path.exists(cf):
    with open(cf) as fh:
        cl = [l.strip() for l in fh if l.strip()]
    print('=== encoder_multiclass_23_classes.txt === count=', len(cl))
    print(cl)
else:
    print('encoder_multiclass_23_classes.txt NOT FOUND')
