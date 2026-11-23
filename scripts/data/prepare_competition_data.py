"""Reconstruct one paper dataset from its competition input file."""
from pathlib import Path
from datetime import datetime
import argparse
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data.hierarchy import load_hierarchy
from src.data.datasets import semantic_values_fingerprint


def walmart(path, names):
    days = [f'd_{i}' for i in range(1, 1942)]
    sales = pd.read_csv(path, usecols=['store_id', 'cat_id', *days])
    totals = sales.groupby(sales.store_id + '_' + sales.cat_id)[days].sum()
    dates = pd.date_range('2011-01-29', periods=len(days), freq='D')
    return dates, totals.loc[names].to_numpy().T


def gefcom(path, names):
    import openpyxl
    dates = pd.date_range('2005-01-01', '2011-12-31 23:00', freq='h')
    values = np.full((len(dates), len(names)), np.nan)
    lookup = {node: i for i, node in enumerate(names)}
    seen = set()
    book = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        rows = iter(book.active.values)
        next(rows)
        for meter, day, *hours in rows:
            node = str(meter)
            if node not in lookup:
                continue
            index = (day - datetime(2005, 1, 1)).days * 24
            if not 0 <= index <= len(dates) - 24:
                continue
            if (node, index) in seen:
                raise ValueError(f'Duplicate meter-day: {node} {day}')
            seen.add((node, index))
            values[index:index+24, lookup[node]] = hours
    finally:
        book.close()
    return dates, pd.DataFrame(values, index=dates).interpolate(method='time').to_numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', choices=['walmart', 'gefcom2017f'])
    parser.add_argument('input', type=Path)
    args = parser.parse_args()
    specs = {
        'walmart': ('Walmart_4level', 'Walmart_processed_without_exog.csv', 'Date', 'df8776c55c771456462955c9340e5f6e', walmart),
        'gefcom2017f': ('Gefcom2017F_4level', 'load_processed.csv', 'timestamp', 'e933f3d0876ac5d5bf654a370d5745e3', gefcom),
    }
    dirname, filename, timestamp, expected, prepare = specs[args.dataset]
    folder = ROOT/'Data'/dirname
    hier = load_hierarchy(folder)
    names = [hier.node_names[i] for i in hier.bottom_idx]
    dates, bottoms = prepare(args.input, names)
    values = bottoms @ hier.S.astype('float64').T
    actual = semantic_values_fingerprint(dates, hier.node_names, values)
    if actual != expected:
        raise ValueError(f'Input differs from paper: {actual} (expected {expected})')
    frame = pd.DataFrame(values, columns=hier.node_names, index=dates)
    frame.to_csv(folder/filename, index_label=timestamp)
    print(f'{dirname}: {len(frame)} timestamps, {len(frame.columns)} nodes, {actual}')


if __name__ == '__main__':
    main()
