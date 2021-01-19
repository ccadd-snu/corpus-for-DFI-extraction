# Manually annotated corpus for DFI extraction

`DFI (Drug-Food interaction) corpus` is the largest manually annotated corpus consisted of 2271 abstracts of biomedical articles published by PubMed for developing an NLP model extracting DFI. 1.	We introduced our manually annotated corups for extracting DFI information from abstracts of biomedical articles and suggested ‘DFI key-sentence’ as a target entity for DFI extraction. To best our knowledge, our dataset for DFI extraction is the first manually annotated dataset for extracting DFI from biomedical articles and the largest and the most comprehensive dataset for extracting drug interaction, including DDI.

<p align="center"><img src= 'https://user-images.githubusercontent.com/75958220/104395745-c1321780-558c-11eb-9121-2fa7895c56ff.png' width='450' height='450'></p>
<center> **Figure 1.** Example of a manually annotated abstract for DFI extraction. </center>

## Distribution of evidence-level and named entities of the `DFI corpus`

**Table 1.** Distribution of the annotated evidence-levels in the `DFI corpus`
|Evidence-level  | Training | Development | Test |
| ------------ | :-----------: | :-----------: | :-----------: |
|'clinical trial'       | 116 (7.30) |	33 (7.24)	| 16 (7.08)
|'observational study'       | 78 (4.91)	| 23 (5.04)	| 11 (4.87)
|'case report'       | 30 (1.89)	| 9 (2.97)	| 4 (1.77) |
|'in-vivo study'       | 547 (34.42)	| 157 (34.42)	| 78 (34.51) |
|'in-vitro study'       | 477 (30.02)	| 137 (30.04)	| 68 (30.09) |
|'bioanalysis'       |  91 (5.73)	| 26 (5.70)	| 13 (5.75) |
|'others'       | 250 (15.73)	| 71 (15.57)	| 36 (15.93) |

**Table 2.** Distribution of the annotated entity types in the `DFI corpus`
|Entity type  | Training | Development | Test |
| ------------ | :-----------: | :-----------: | :-----------: |
|'drug'	|	5632 (1.46)	|	1669 (1.48)	|	787 (1.43)	|
|'food'	|	9384 (2.44)	|	2621 (2.33)	|	1348 (2.45)	|
|'food component'	|	902 (0.23)	|	377 (0.34)	|	63 (0.11)	|
|'ambiguous'	|	452 (0.12)	|	118 (0.10)	|	153 (0.28)	|
|'well known target'	|	6065 (1.58)	|	1723 (1.53)	|	679 (1.24)	|
|'drug metabolizer'	|	697 (0.18)	|	176 (0.16)	|	125 (0.23)	|
|'drug transporter'	|	288 (0.07)	|	113 (0.10)	|	14 (0.03)	|
|'none'	|	361545 (93.92)	|	105688 (93.96)	|	51752 (94.23)	|
|total	|	384965 (100.0)	|	112485 (100.0)	|	54921 (100.0)	|


