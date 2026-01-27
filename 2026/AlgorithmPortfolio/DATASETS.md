# Mesa Scale-Free SIR — Datasets

## Synthetic (default)
- Barabási–Albert graphs
- Erdős–Rényi graphs

## Optional validation
- SocioPatterns contact networks: https://www.sociopatterns.org/datasets.html

### Canonical dataset citations
If you use the SocioPatterns contact networks in a paper, please cite the corresponding dataset papers:
- HS2013: Mastrandrea et al. (2015), DOI: 10.1371/journal.pone.0136497
- HighSchool2012: Fournet \& Barrat (2014), DOI: 10.1371/journal.pone.0107878
- PrimarySchool: Stehl{\'e} et al. (2011), DOI: 10.1371/journal.pone.0023176
- HospitalLyon: Vanhems et al. (2013), DOI: 10.1371/journal.pone.0073970
- WorkplaceInVS15: G{\'e}nois \& Barrat (2018), DOI: 10.1140/epjds/s13688-018-0140-1

### Quickstart (recommended SocioPatterns contact lists)
1. Download:
   - `mkdir -p data/raw/sociopatterns`
   - `curl -L -o data/raw/sociopatterns/HighSchool2013_proximity_net.csv.gz https://www.sociopatterns.org/assets/data/HighSchool2013_proximity_net.csv.gz`
   - `curl -L -o data/raw/sociopatterns/highschool_2012.csv.gz https://www.sociopatterns.org/assets/data/highschool_2012.csv.gz`
   - `curl -L -o data/raw/sociopatterns/primaryschool.csv.gz https://www.sociopatterns.org/assets/data/primaryschool.csv.gz`
   - `curl -L -o data/raw/sociopatterns/hospital_lyon_contacts.dat.gz https://www.sociopatterns.org/assets/data/hospital_lyon_contacts.dat.gz`
   - `curl -L -o data/raw/sociopatterns/workplace_InVS15_tij.dat.gz https://www.sociopatterns.org/assets/data/workplace_InVS15_tij.dat.gz`
2. Run an intervention benchmark including the real network:
   - Set `real_networks:` in your YAML config (see `src/config_transfer.yaml`).

Notes:
- The loader treats SocioPatterns “contact list” files as whitespace-separated and uses the first 3 columns (`t`, `i`, `j`), ignoring any extra columns.
- SocioPatterns data are distributed by the SocioPatterns collaboration; please cite them appropriately and follow their terms of use when redistributing derived artifacts.
