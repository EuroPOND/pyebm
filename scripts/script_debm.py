# *=========================================================================
# *
# *  Copyright Erasmus MC Rotterdam and contributors
# *
# *  Licensed under the GNU GENERAL PUBLIC LICENSE Version 3;
# *  you may not use this file except in compliance with the License.
# *
# *  Unless required by applicable law or agreed to in writing, software
# *  distributed under the License is distributed on an "AS IS" BASIS,
# *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# *  See the License for the specific language governing permissions and
# *  limitations under the License.
# *
# *=========================================================================*/

## Simplest function call (with default parameters)
from __future__ import print_function

from pyebm import debm, ebm

ModelOutput, SubjTrain, SubjTest = debm.fit('./resources/Data_7.csv')
print([ModelOutput.BiomarkerList[x] for x in ModelOutput.MeanCentralOrdering])

## Example with Visual Biomarker Distributions as output

from collections import namedtuple

MO = namedtuple('MethodOptions', 'MixtureModel Bootstrap')
MO.Bootstrap = 0;
MO.MixtureModel = 'vv2';
VO = namedtuple('VerboseOptions', 'Distributions')
VO.Distributions = 1;
ModelOutput, SubjTrain, SubjTest = debm.fit('./resources/Data_7.csv', MethodOptions=MO, VerboseOptions=VO)

## Example with bootstrapping and visual output

from collections import namedtuple

MO = namedtuple('MethodOptions', 'MixtureModel Bootstrap')
MO.Bootstrap = 5;
MO.MixtureModel = 'vv2';
VO = namedtuple('VerboseOptions', 'Ordering PlotOrder Distributions')
VO.Ordering = 1;
VO.PlotOrder = 1;
VO.Distributions = 0;
ModelOutput, SubjTrain, SubjTest = debm.fit('./resources/Data_7.csv', MethodOptions=MO, VerboseOptions=VO)

print([ModelOutput.BiomarkerList[x] for x in ModelOutput.MeanCentralOrdering])

## Example with Patient Staging and visual output. Also, a pandas dataframe can be sent as an input instead of CSV
from collections import namedtuple

MO = namedtuple('MethodOptions', 'MixtureModel Bootstrap PatientStaging')
MO.Bootstrap = 0;
MO.MixtureModel = 'vv2';
MO.PatientStaging = ['ml', 'l']
VO = namedtuple('VerboseOptions', 'Distributions PatientStaging')
VO.PatientStaging = 1;
VO.Distributions = 0;
import pandas as pd

D = pd.read_csv('./resources/Data_7.csv')
ModelOutput, SubjTrain, SubjTest = debm.fit(D, MethodOptions=MO, VerboseOptions=VO)

from collections import namedtuple

MO = namedtuple('MethodOptions', 'MixtureModel Bootstrap PatientStaging')
MO.Bootstrap = 0;
MO.MixtureModel = 'vv2';
MO.PatientStaging = ['exp', 'p']
VO = namedtuple('VerboseOptions', 'Distributions PatientStaging')
VO.PatientStaging = 1;
VO.Distributions = 0;
import pandas as pd

D = pd.read_csv('./resources/Data_7.csv')
ModelOutput, SubjTrain, SubjTest = debm.fit(D, MethodOptions=MO, VerboseOptions=VO)

## Comparing AUCs of Patient Staging using Cross-Validation with Training and Testset
## Comparison will be done between DEBM / EBM / SVM

from collections import namedtuple

MO1 = namedtuple('MethodOptions', 'MixtureModel Bootstrap PatientStaging')
MO1.Bootstrap = 0;
MO1.MixtureModel = 'vv2';
MO1.PatientStaging = ['exp', 'p']

MO2 = namedtuple('MethodOptions', 'MixtureModel Bootstrap PatientStaging')
MO2.Bootstrap = 0;
MO2.MixtureModel = 'vv2';
MO2.PatientStaging = ['ml', 'l']

VO = namedtuple('VerboseOptions', 'Distributions PatientStaging')
VO.PatientStaging = 0;
VO.Distributions = 0;
import pandas as pd

D = pd.read_csv('./resources/Data_7.csv');
Y = D['Diagnosis'].copy();
Y[Y == 'CN'] = 0;
Y[Y == 'AD'] = 2;
Y[Y == 'MCI'] = 1;

from sklearn.model_selection import KFold as KF
from sklearn import metrics
import numpy as np

skf = KF(n_splits=10, shuffle=True, random_state=42)
print("Comparing the AUCs of CN / AD Classification:")
print("Cross-Validation Iteration:")
auc1 = [];
auc2 = [];

count = -1
for train_index, test_index in skf.split(D, pd.to_numeric(Y.values)):
    count = count + 1;
    print([count],end="")
    DTrain, DTest = D.iloc[train_index], D.iloc[test_index]
    ModelOutput1, SubjTrain1, SubjTest1 = debm.fit(DTrain, MethodOptions=MO1, VerboseOptions=VO, DataTest=DTest)
    ModelOutput2, SubjTrain2, SubjTest2 = ebm.fit(DTrain, MethodOptions=MO2, VerboseOptions=VO, DataTest=DTest)
    Y = DTest['Diagnosis']
    idx = Y != 'MCI';
    Y = Y[idx];
    Y[Y == 'CN'] = 0;
    Y[Y == 'AD'] = 1;

    S = SubjTest1[0]['Stages'];
    S = S.values[idx];
    auc1.append(metrics.roc_auc_score(pd.to_numeric(Y.values), S))
    S = SubjTest2[0]['Stages'];
    S = S.values[idx];
    auc2.append(metrics.roc_auc_score(pd.to_numeric(Y.values), S))

print("\nMean AUC using DEBM with Patient Staging Option:", MO1.PatientStaging, '--->', np.mean(auc1))
print("Mean AUC using EBM with Patient Staging Option:", MO2.PatientStaging, '--->', np.mean(auc2))
