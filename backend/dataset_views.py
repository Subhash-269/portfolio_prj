import os
import pandas as pd
from django.http import JsonResponse
from rest_framework.decorators import api_view
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi

CSV_PATHS = [
    os.path.join("Training", "sp500_companies.csv"),
    "sp500_companies.csv"
]
MAPPER_PATHS = [
    os.path.join("Training", "mapper.json"),
    "mapper.json"
]

def _load_sp500_csv():
    for p in CSV_PATHS:
        if os.path.exists(p):
            return pd.read_csv(p)
    return None

def _load_mapper_path():
    # return first writable/usable path
    for p in MAPPER_PATHS:
        base = os.path.dirname(p)
        if base and not os.path.exists(base):
            try:
                os.makedirs(base, exist_ok=True)
            except Exception:
                continue
        return p
    return "mapper.json"

# Single API: return unique GICS Sector from mapper
gics_list_schema = openapi.Schema(
    type=openapi.TYPE_OBJECT,
    properties={
        'values': openapi.Schema(
            type=openapi.TYPE_ARRAY,
            items=openapi.Schema(type=openapi.TYPE_STRING),
            description='Sorted unique GICS sectors from mapper.json'
        )
    }
)

@swagger_auto_schema(
    method='get',
    operation_description='Get unique GICS Sector values. Uses mapper.json if present; otherwise builds mapper from sp500_companies.csv and saves it.',
    responses={200: gics_list_schema, 500: 'Server Error'}
)
@api_view(['GET'])
def sp500_gics_sectors(request):
    import json
    mapper_path = _load_mapper_path()
    mapper = None
    # Try load mapper
    if os.path.exists(mapper_path):
        try:
            with open(mapper_path, 'r', encoding='utf-8') as f:
                mapper = json.load(f)
        except Exception:
            mapper = None
    # Build mapper if missing
    if mapper is None:
        df = _load_sp500_csv()
        if df is None:
            return JsonResponse({'error': 'sp500_companies.csv not found'}, status=500)
        # Detect columns
        sym_col = None
        for c in ['Symbol', 'Ticker', 'symbol', 'SYMBOL']:
            if c in df.columns:
                sym_col = c
                break
        gics_col = None
        for c in ['GICS Sector', 'Sector', 'GICS_Sector', 'gics_sector']:
            if c in df.columns:
                gics_col = c
                break
        abbr_col = None
        for c in ['Security', 'security', 'SECURITY']:
            if c in df.columns:
                abbr_col = c
                break
        if sym_col is None or gics_col is None:
            return JsonResponse({'error': 'Required columns not found in CSV'}, status=500)
        # Build mapping Symbol -> GICS Sector
        mapper = {}
        use_cols = [sym_col, gics_col] + ([abbr_col] if abbr_col else [])
        for _, row in df[use_cols].dropna().iterrows():
            sym = str(row[sym_col]).strip()
            gics = str(row[gics_col]).strip()
            abbr = str(row[abbr_col]).strip() if abbr_col else ''
            if sym and gics:
                mapper[sym] = {'sector': gics, 'abbr': abbr}
        # Save mapper
        try:
            with open(mapper_path, 'w', encoding='utf-8') as f:
                json.dump(mapper, f, indent=2)
        except Exception:
            # non-fatal; proceed without saving
            pass
    # Return unique sectors
    def _sector_of(val):
        if isinstance(val, dict):
            return str(val.get('sector', '')).strip()
        return str(val).strip()
    values = sorted(set(_sector_of(v) for v in mapper.values() if _sector_of(v)))
    return JsonResponse({'values': values})

@swagger_auto_schema(
    method='get',
    operation_description='Compute 1Y/3Y/5Y equal-weight returns for each GICS Sector using diversified_market_data.csv',
    responses={200: openapi.Schema(type=openapi.TYPE_ARRAY, items=openapi.Items(type=openapi.TYPE_OBJECT))}
)
@api_view(['GET'])
def gics_sector_returns(request):
    import json
    # Load mapper
    mapper_path = _load_mapper_path()
    if not os.path.exists(mapper_path):
        # attempt to build mapper if absent
        _ = sp500_gics_sectors(request)
    mapper = {}
    try:
        with open(mapper_path, 'r', encoding='utf-8') as f:
            mapper = json.load(f)
    except Exception:
        return JsonResponse({'error': 'mapper.json not available'}, status=500)

    # Load market data
    data_paths = [os.path.join('Training', 'diversified_market_data.csv'), 'diversified_market_data.csv']
    csv_path = None
    for p in data_paths:
        if os.path.exists(p):
            csv_path = p
            break
    if csv_path is None:
        return JsonResponse({'error': 'diversified_market_data.csv not found'}, status=500)

    import pandas as pd
    df = pd.read_csv(csv_path)
    if 'Date' not in df.columns or 'Ticker' not in df.columns or 'Close' not in df.columns:
        return JsonResponse({'error': 'CSV must contain Date, Ticker, Close'}, status=500)
    df['Date'] = pd.to_datetime(df['Date'])

    # Keep only tickers present in mapper
    df = df[df['Ticker'].isin(mapper.keys())]
    if df.empty:
        return JsonResponse({'error': 'No overlapping tickers between mapper and data'}, status=500)

    # Build pivot of Close
    pivot = df.pivot(index='Date', columns='Ticker', values='Close').sort_index().ffill().bfill()
    dates = pivot.index
    if len(dates) < 1261:
        # still compute with available data
        pass

    def period_return(days):
        if len(dates) <= days:
            start_idx = 0
        else:
            start_idx = len(dates) - days - 1
        start = pivot.iloc[start_idx]
        end = pivot.iloc[-1]
        ret = (end / start) - 1.0
        return ret

    ret_1y = period_return(252)
    ret_3y = period_return(756)
    ret_5y = period_return(1260)

    # Aggregate equal-weight by sector
    from collections import defaultdict
    sector_to_tickers = defaultdict(list)
    for sym, val in mapper.items():
        sector = val.get('sector') if isinstance(val, dict) else val
        if sym in pivot.columns and sector:
            sector_to_tickers[sector].append(sym)

    results = []
    for sector, tickers in sector_to_tickers.items():
        if not tickers:
            continue
        y1 = float(ret_1y[tickers].mean()) * 100.0
        y3 = float(ret_3y[tickers].mean()) * 100.0
        y5 = float(ret_5y[tickers].mean()) * 100.0
        results.append({'sector': sector, 'y1': round(y1, 2), 'y3': round(y3, 2), 'y5': round(y5, 2)})

    # Sort by 1Y descending for convenience
    results.sort(key=lambda x: x['y1'], reverse=True)
    return JsonResponse(results, safe=False)

@swagger_auto_schema(
    method='get',
    operation_description='Return monthly equal-weight sector index time series from diversified_market_data.csv. Optionally filter by sectors via ?sectors=Tech,Finance',
    responses={200: openapi.Schema(type=openapi.TYPE_OBJECT)}
)
@api_view(['GET'])
def gics_sector_timeseries(request):
    import json
    # Load mapper
    mapper_path = _load_mapper_path()
    if not os.path.exists(mapper_path):
        _ = sp500_gics_sectors(request)
    try:
        with open(mapper_path, 'r', encoding='utf-8') as f:
            mapper = json.load(f)
    except Exception:
        return JsonResponse({'error': 'mapper.json not available'}, status=500)

    # Load market data
    data_paths = [os.path.join('Training', 'diversified_market_data.csv'), 'diversified_market_data.csv']
    csv_path = next((p for p in data_paths if os.path.exists(p)), None)
    if csv_path is None:
        return JsonResponse({'error': 'diversified_market_data.csv not found'}, status=500)

    df = pd.read_csv(csv_path)
    if 'Date' not in df.columns or 'Ticker' not in df.columns or 'Close' not in df.columns:
        return JsonResponse({'error': 'CSV must contain Date, Ticker, Close'}, status=500)
    df['Date'] = pd.to_datetime(df['Date'])

    # Restrict to mapper tickers
    df = df[df['Ticker'].isin(mapper.keys())]
    if df.empty:
        return JsonResponse({'error': 'No overlapping tickers between mapper and data'}, status=500)

    # Build sector -> tickers map
    from collections import defaultdict
    sector_to_tickers = defaultdict(list)
    for sym, val in mapper.items():
        sector = val.get('sector') if isinstance(val, dict) else val
        if sector:
            sector_to_tickers[sector].append(sym)

    # Optional filter
    query = request.GET.get('sectors')
    wanted = None
    if query:
        wanted = {s.strip().lower() for s in query.split(',') if s.strip()}

    # Optional limit (number of points) or 'all'
    limit_param = request.GET.get('limit', '60')
    limit = None
    if isinstance(limit_param, str) and limit_param.lower() == 'all':
        limit = None
    else:
        try:
            limit = max(1, int(limit_param))
        except Exception:
            limit = 60

    # Pivot close prices
    pivot = df.pivot(index='Date', columns='Ticker', values='Close').sort_index().ffill().bfill()

    # Monthly resample (end of month)
    monthly = pivot.resample('M').last()

    # Compute equal-weight sector index normalized to 100 at start
    series = {}
    for sector, tickers in sector_to_tickers.items():
        if not tickers:
            continue
        if wanted and sector.lower() not in wanted:
            continue
        cols = [t for t in tickers if t in monthly.columns]
        if len(cols) == 0:
            continue
        # Equal-weight portfolio value: average of normalized prices
        sub = monthly[cols]
        base = sub.iloc[0]
        base[base == 0] = 1.0
        norm = sub.div(base).mean(axis=1) * 100.0
        # Downsample if limit is set
        if limit is not None and len(norm) > limit:
            norm = norm.iloc[-limit:]
        series[sector] = [{'date': d.strftime('%Y-%m-%d'), 'value': round(float(v), 2)} for d, v in norm.items()]

    return JsonResponse({'series': series}, safe=False)

@swagger_auto_schema(
    method='get',
    operation_description='Compute 1Y/3Y/5Y equal-weight returns for each Commodity Type using Training/Commodities/commodities.csv',
    responses={200: openapi.Schema(type=openapi.TYPE_ARRAY, items=openapi.Items(type=openapi.TYPE_OBJECT))}
)
@api_view(['GET'])
def commodities_returns(request):
    # Locate commodities CSV
    data_paths = [
        os.path.join('Training', 'Commodities', 'commodities.csv'),
        os.path.join('Commodities', 'commodities.csv'),
        'commodities.csv'
    ]
    csv_path = next((p for p in data_paths if os.path.exists(p)), None)
    if csv_path is None:
        return JsonResponse({'error': 'Commodities CSV not found'}, status=500)

    df = pd.read_csv(csv_path)
    # Expect columns: Date, Commodity Type, Close
    if 'Date' not in df.columns or 'Close' not in df.columns:
        return JsonResponse({'error': 'CSV must contain Date and Close'}, status=500)
    # Normalize commodity identifier to Ticker-like key
    commodity_col = None
    for c in ['Commodity Type', 'Commodity', 'Type']:
        if c in df.columns:
            commodity_col = c
            break
    if commodity_col is None:
        return JsonResponse({'error': 'CSV must contain Commodity Type column'}, status=500)

    df['Date'] = pd.to_datetime(df['Date'])

    # Pivot close prices per commodity type
    df = df.rename(columns={commodity_col: 'Commodity'})
    pivot = df.pivot(index='Date', columns='Commodity', values='Close').sort_index().ffill().bfill()
    dates = pivot.index
    if len(dates) == 0:
        return JsonResponse({'error': 'No data points available'}, status=500)

    def period_return(days):
        if len(dates) <= days:
            start_idx = 0
        else:
            start_idx = len(dates) - days - 1
        start = pivot.iloc[start_idx]
        end = pivot.iloc[-1]
        ret = (end / start) - 1.0
        return ret

    ret_1y = period_return(252)
    ret_3y = period_return(756)
    ret_5y = period_return(1260)

    results = []
    for commodity in pivot.columns:
        y1 = float(ret_1y.get(commodity, 0.0)) * 100.0
        y3 = float(ret_3y.get(commodity, 0.0)) * 100.0
        y5 = float(ret_5y.get(commodity, 0.0)) * 100.0
        results.append({'asset': commodity, 'y1': round(y1, 2), 'y3': round(y3, 2), 'y5': round(y5, 2)})

    # Sort by 1Y descending
    results.sort(key=lambda x: x['y1'], reverse=True)
    return JsonResponse(results, safe=False)
