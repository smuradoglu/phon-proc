# phon-proc

Data and code for [Do transformer models do phonology like a linguist?](https://aclanthology.org/2023.findings-acl.541) (Muradoglu & Hulden, Findings 2023)

1. Generate lexicon by running lexicon_generation.py
2. Import lexicon into phonological_processes.lexc
3. Generate input/output pairs for each phonological process using the FST defined by phonological_processes.lexc and phonological_processes.foma
4. Sample input/output pairs according to experiment design. 

The generated input/output pairs and data used for reported results can be found in the data folder. \

Notation: \
Apha - Apharesis \
Apoc - Apocope \
all_proc - All 29 phonological processes \
Wfdevoice - Word-final devoicing \
WIdevoice - Word-initial devoicing \
WIvoice - Word-initial voicing \
IntVoice - Intervocalic voicing \
Degem - Degemmination \
Gem - Gemmination \
Vshort - Vowel Shortening \
Vlength - Vowel lengthening \
Haplo - Haplology \

