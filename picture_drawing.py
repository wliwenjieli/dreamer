import pandas as pd
import numpy as np
import os
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

awake_phase=pd.read_csv("awake_phase.csv")
dream_phase=pd.read_csv("dream_phase.csv",index_col=0)


phase1=awake_phase[:100]
phase2=awake_phase[100:105]
phase3=dream_phase
phase4=awake_phase[105:]

# fear_consolidation=pd.concat([phase1,phase2,phase3,phase4],sort=False)
# fear_consolidation=fear_consolidation.reset_index(drop=True)
# a=np.array(range(len(fear_consolidation)))
# fear_consolidation=fear_consolidation.set_index([["phase1"]*100+["phase2"]*5+["phase3"]*len(dream_phase)+["phase4"]*5,a])
# fear_consolidation.to_csv("fear_consolidation.csv",index=True)

fear_consolidation=pd.concat([phase1,phase2,phase3,phase4],sort=False)
fear_consolidation=fear_consolidation.reset_index(drop=True)
a=np.array(range(len(fear_consolidation)))
fear_consolidation=fear_consolidation.set_index([["phase1"]*100+["phase2"]*5+["phase3"]*len(dream_phase)+["phase4"]*5,a])
fear_consolidation.to_csv("fear_extinction.csv",index=True)