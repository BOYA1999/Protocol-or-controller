import csv
import hashlib
import io
import json
import math
import urllib.request
import zipfile
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data' / 'marginal'
OUT.mkdir(parents=True, exist_ok=True)
COMMIT = '599a970669daf77f8409ce970aca79e9f03d9047'
GITHUB = f'https://raw.githubusercontent.com/citylearn-project/CityLearn/{COMMIT}/'
SOURCES = {}
checkpoint = OUT / 'build_summary.json'
if not checkpoint.exists():
    checkpoint = ROOT / 'outputs' / 'marginal_carbon_sources.json'
FROZEN = json.loads(checkpoint.read_text(encoding='utf-8')) if checkpoint.exists() else {}


def verify_frozen(name, data, section):
    expected = FROZEN.get(section, {}).get(name, {}).get('sha256')
    if expected is not None and hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f'Frozen-data mismatch: {name}; existing outputs were not replaced.')


def source(name, url):
    path = OUT / name
    if not path.exists():
        request = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        path.write_bytes(urllib.request.urlopen(request, timeout=120).read())
    data = path.read_bytes()
    SOURCES[name] = dict(url=url, bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    return data


def write_csv(name, columns, rows):
    handle = io.StringIO(newline='')
    writer = csv.writer(handle, lineterminator='\n')
    writer.writerow(columns)
    writer.writerows(rows)
    data = handle.getvalue().encode('utf-8')
    verify_frozen(name, data, 'processed_files')
    (OUT / name).write_bytes(data)


archive = source('SGIP_CAISO_SCE.zip', 'https://data.sgipsignal.com/historical/SGIP_CAISO_SCE.zip')
api = json.loads(source('sgip-openapi.json', 'https://docs.sgipsignal.com/sgip-openapi.json'))
notebook = json.loads(source('citylearn_frozen_tutorial.ipynb', GITHUB + 'examples/tutorial.ipynb'))
original_schema = json.loads(source('citylearn_phase_all_schema.json',
    GITHUB + 'data/datasets/citylearn_challenge_2022_phase_all/schema.json'))
benchmark_schema = json.loads((ROOT / 'data' / 'citylearn2022' / 'schema.json').read_text())
source('sgip_download_data.html', 'https://content.sgipsignal.com/download-data/')
source('sgip_grid_regions.html', 'https://content.sgipsignal.com/grid-regions/')
unit_evidence = api['paths']['/sgipmoer']['get']['description']
assert 'kg CO2 per kWh' in unit_evidence
date_evidence = next(''.join(cell['source']) for cell in notebook['cells']
                     if 'August 1, 2016' in ''.join(cell.get('source', [])))
assert 'July 31, 2017' in date_evidence
versions_evidence = next(tag['description'] for tag in api['tags']
                         if 'Starting January 1, 2022' in tag.get('description', ''))

records = []
raw_files = {}
with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
    for month in range(1, 9):
        name = f'SGIP_CAISO_SCE_2017-{month:02}_ALL_MOER_VERSIONS.csv'
        data = bundle.read(name)
        verify_frozen(name, data, 'extracted_files')
        (OUT / name).write_bytes(data)
        raw_files[name] = dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
        for row in csv.DictReader(io.StringIO(data.decode('utf-8-sig'))):
            stamp = datetime.fromisoformat(row['timestamp'])
            assert stamp.utcoffset() == timedelta(0)
            values = tuple(float(row[f'MOER version {version}.0']) for version in (2, 1))
            assert all(math.isfinite(value) for value in values)
            assert stamp.minute % 5 == 0 and stamp.second == 0
            records.append((stamp, *values))
records.sort()
assert len(records) == 69984
assert all(right[0] - left[0] == timedelta(minutes=5)
           for left, right in zip(records, records[1:]))
write_csv('sce_moer_2017_5min.csv',
          ['timestamp_utc', 'moer_v2_kg_co2_per_kwh', 'moer_v1_kg_co2_per_kwh'],
          [(stamp.isoformat(), v2, v1) for stamp, v2, v1 in records])
groups = defaultdict(list)
for stamp, v2, v1 in records:
    groups[stamp.replace(minute=0)].append((v2, v1))
assert all(len(values) == 12 for values in groups.values())
pst, pacific = timezone(timedelta(hours=-8)), ZoneInfo('America/Los_Angeles')
hourly = [(stamp, sum(x[0] for x in values) / 12, sum(x[1] for x in values) / 12)
          for stamp, values in sorted(groups.items())]
anchor = datetime(2017, 1, 1, tzinfo=pst)
required = {(anchor + timedelta(hours=i)).astimezone(timezone.utc) for i in range(-1, 5089)}
assert required.issubset({row[0] for row in hourly})
write_csv('sce_moer_2017_hourly.csv',
          ['timestamp_utc', 'timestamp_pst', 'timestamp_pacific_civil',
           'moer_v2_kg_co2_per_kwh', 'moer_v1_kg_co2_per_kwh'],
          [(stamp.isoformat(), stamp.astimezone(pst).isoformat(),
            stamp.astimezone(pacific).isoformat(), v2, v1) for stamp, v2, v1 in hourly])

comparisons = []
for building in range(1, 18):
    name = f'Building_{building}.csv'
    url = GITHUB + 'data/datasets/citylearn_challenge_2022_phase_all/' + name
    upstream = urllib.request.urlopen(url, timeout=60).read()
    a = list(csv.DictReader(io.StringIO(upstream.decode())))
    local = (ROOT / 'data' / 'citylearn2022' / name).read_bytes()
    b = list(csv.DictReader(io.StringIO(local.decode())))
    differences = {column: sum(x[column] != y[column] for x, y in zip(a, b))
                   for column in ('non_shiftable_load', 'solar_generation')}
    assert len(a) == len(b) == 8760 and not any(differences.values())
    for i, row in enumerate(b):
        stamp = datetime(2016, 8, 1) + timedelta(hours=i)
        midnight = i > 0 and stamp.hour == 0
        display = stamp - timedelta(hours=1) if midnight else stamp
        assert (int(row['month']), int(row['hour']), int(row['day_type'])) == (
            display.month, 24 if midnight else stamp.hour, display.isoweekday())
    comparisons.append(dict(building=building, url=url, rows=len(a), differences=differences,
                            calendar_mismatches_after_hour24_interpretation=0,
                            phase_all_pv_kwp=original_schema['buildings'][name[:-4]]['pv']['attributes']['nominal_power'],
                            benchmark_pv_kwp=benchmark_schema['buildings'][name[:-4]]['pv']['attributes']['nominal_power'],
                            upstream_sha256=hashlib.sha256(upstream).hexdigest(),
                            local_sha256=hashlib.sha256(local).hexdigest()))

summary = dict(
    analysis='Post hoc retrospective marginal operating emission-rate re-evaluation',
    region='CAISO Southern California Edison DLAP (SGIP_CAISO_SCE)',
    units='kg CO2/kWh',
    resolution='5-minute source; hourly arithmetic mean of exactly 12 observations',
    source_files=SOURCES, extracted_files=raw_files,
    reproduction_gate='Existing checkpoint SHA256 values for eight monthly CSVs and processed CSVs must match before replacement; the growing source ZIP is recorded but not locked.',
    utc_coverage=[records[0][0].isoformat(), records[-1][0].isoformat()],
    five_minute_rows=len(records), hourly_rows=len(hourly),
    duplicates=0, gaps_over_5min=0, incomplete_hours=0,
    requested_5088_hour_overlap_plus_1h_edges_complete=True,
    five_minute_statistics={f'v{version}': dict(min=min(row[index] for row in records),
        max=max(row[index] for row in records),
        mean=sum(row[index] for row in records)/len(records),
        negative_count=sum(row[index] < 0 for row in records))
        for version, index in ((2, 1), (1, 2))},
    source_evidence=dict(unit_description=unit_evidence, model_versions=versions_evidence,
                         citylearn_date_description=date_evidence),
    citylearn_profile_comparisons=comparisons,
    alignment=dict(citylearn_load_pv_period='2016-08-01 through 2017-07-31, from official tutorial',
        supported_overlap='2017-01-01 through 2017-07-31 (5,088 hours)',
        citylearn_indices='3672:8760, zero-based Python slicing',
        fixed_pst='UTC-08:00; analysis convention, source timezone not independently established',
        alternative='America/Los_Angeles civil clock, and plus/minus one-hour phase sensitivities',
        calendar_ambiguity='Hour 24 carries preceding-day calendar fields; first plus_evs row is changed to hour 0. All daylight_savings_status values are 0.',
        schema_date='Local plus_evs schema start_date is 2024-08-01; it is not evidence of collection year.',
        price='Frozen plus_evs price differs from phase_all TOU; retain as benchmark signal, not historical SCE tariff.'),
    limits=['2017 model-version series are retrospective backcasts, not forecasts available to a controller in 2017.',
        'CityLearn operational grid-intensity factors are supplied by EPRI; retrieved metadata does not establish average versus marginal methodology.',
        'MOER estimates support modeled operational counterfactual changes, not field-measured avoided emissions.',
        'Only normalized solar_generation profiles match phase_all; plus_evs nominal PV capacities differ in seven buildings, so benchmark PV power is scaled rather than a historical measured system.',
        'Use net exchange changes relative to no battery for consequential accounting; gross-import weighting omits changes in displaced exports.',
        'The seven-month overlap must not be annualized as a verified full-year marginal-carbon result.'],
    access=dict(official_statement='Public zipped CSV download for reference and analysis purposes; no authentication needed.',
                license='No explicit open redistribution licence located; preserve attribution and source links; do not infer CC BY or MIT for SGIP data.'))
for filename in ('sce_moer_2017_5min.csv', 'sce_moer_2017_hourly.csv'):
    data = (OUT / filename).read_bytes()
    summary.setdefault('processed_files', {})[filename] = dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
(OUT / 'build_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
(ROOT / 'outputs' / 'marginal_carbon_sources.json').write_text(
    json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
print(json.dumps({key: summary[key] for key in ('five_minute_rows', 'hourly_rows', 'utc_coverage',
                                               'five_minute_statistics', 'processed_files')}, indent=2))
