import React from 'react';
import { ResponsiveContainer, PieChart, Pie, Cell, Tooltip, Legend, LineChart, Line, XAxis, YAxis, CartesianGrid, ReferenceLine } from 'recharts';
// Custom legend enabling hover highlight
function HoverLegend({ payload, onHover, onLeave }) {
  if (!payload) return null;
  return (
    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.75rem', marginTop: '0.25rem' }}>
      {payload.map((item) => (
        <div
          key={item.value}
          onMouseEnter={() => onHover && onHover(item.value)}
          onMouseLeave={() => onLeave && onLeave()}
          style={{ display: 'inline-flex', alignItems: 'center', gap: '0.4rem', cursor: 'pointer', color: '#374151', fontSize: '0.85rem' }}
        >
          <span style={{ width: 10, height: 10, borderRadius: '50%', background: item.color, display: 'inline-block' }}></span>
          <span>{item.value}</span>
        </div>
      ))}
    </div>
  );
}
// Compact tooltip for multi-series line chart
function CompactTooltip({ active, payload, label }) {
  if (!active || !payload || payload.length === 0) return null;
  const dateStr = (() => { try { return new Date(label).toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: '2-digit' }); } catch { return label; } })();
  return (
    <div style={{ background: '#fff', border: '1px solid #e5e7eb', borderRadius: '0.5rem', boxShadow: '0 2px 8px rgba(30,41,59,0.08)', padding: '0.5rem 0.6rem', fontSize: '0.85rem', color: '#111827' }}>
      <div style={{ marginBottom: '0.25rem', fontWeight: 600 }}>{dateStr}</div>
      <div style={{ display: 'grid', gap: '0.2rem' }}>
        {payload.map((p) => (
          <div key={p.dataKey} style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
            <span style={{ width: 10, height: 10, borderRadius: '50%', background: p.color, display: 'inline-block' }}></span>
            <span style={{ color: '#374151' }}>{p.dataKey}:</span>
            <span style={{ marginLeft: 'auto', color: '#111827', fontWeight: 600 }}>${Math.round(p.value)}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

const assetPerformanceData = [
  // { asset: 'Gold', y1: 57.94, y3: 33.02, y5: 18.40 },
  // { asset: 'Intl Developed Markets', y1: 26.26, y3: 15.87, y5: 9.49 },
  // { asset: 'Emerging Markets', y1: 22.81, y3: 14.30, y5: 6.22 },
  { asset: 'US Stock Market (S&P 500)', y1: 17.53, y3: 24.89, y5: 16.45 },
  // { asset: 'Commodities', y1: 15.84, y3: 4.26, y5: 11.62 },
  // { asset: 'Intermediate Treasuries', y1: 6.44, y3: 4.17, y5: -0.18 },
  // { asset: 'Total Bond Market', y1: 5.70, y3: 4.52, y5: -0.34 },
  // { asset: 'Short Treasuries', y1: 5.09, y3: 4.44, y5: 1.68 },
  { asset: 'US Bonds', y1: 3.74, y3: 1.96, y5: -4.43 },
  // { asset: 'Long Treasuries', y1: 1.71, y3: 0.59, y5: -7.15 },
];

function DashboardHeader() {
  return (
    <header className="dashboard-header" style={{ position: 'sticky', top: 0, zIndex: 1000, background: '#f3f4f6', paddingTop: '0.75rem' }}>
      <div className="dashboard-title" style={{ color: '#22223b', fontWeight: 700, fontSize: '1.8rem', letterSpacing: '0.2px' }}>
        EAI 6050: AI Portfolio
      </div>
      {/* <nav className="dashboard-nav" style={{ display: 'flex', gap: '2rem', fontSize: '1rem', color: '#22223b' }}>
        <a href="#">Analysis</a>
        <a href="#">Markets</a>
        <a href="#">Docs</a>
        <a href="#">Region</a>
        <a href="#">Tools</a>
        <a href="#">Sign Up</a>
        <a href="#">Log In</a>
      </nav> */}
    </header>
  );
}

function SegmentedControl({ segments, active, onChange }) {
  return (
    <div className="segmented-control" style={{ position: 'sticky', top: 56, zIndex: 999, display: 'flex', gap: '0.75rem', marginBottom: '1.25rem', justifyContent: 'center', background: '#f3f4f6', padding: '0.5rem 0' }}>
      {segments.map((seg, idx) => (
        <button
          key={seg}
          className={`segment-btn${active === idx ? ' active' : ''}`}
          style={{
            background: active === idx ? '#22223b' : '#f3f4f6',
            color: active === idx ? '#fff' : '#22223b',
            border: 'none',
            borderRadius: '1.25rem',
            padding: '0.6rem 1.2rem',
            fontSize: '0.95rem',
            fontWeight: 500,
            boxShadow: '0 1px 4px rgba(30,41,59,0.04)',
            cursor: 'pointer',
            transition: 'background 0.2s, color 0.2s',
          }}
          onClick={() => onChange(idx)}
        >
          {seg}
        </button>
      ))}
    </div>
  );
}

function PortfolioResults({ result }) {
  const hasApi = result && Array.isArray(result.portfolio);
  const sectors = hasApi ? result.sectors || [] : [];
  const tickersUsed = hasApi ? result.tickers_used || [] : [];
  const assetsUsed = hasApi ? result.assets_used || [] : [];
  const companiesUsedCount = hasApi ? (Array.isArray(result.tickers_used) ? result.tickers_used.length : 0) : 0;
  const portfolio = hasApi ? [...result.portfolio] : [];
  const [showAll, setShowAll] = React.useState(false);
  const [sortKey, setSortKey] = React.useState('allocation');
  const [sortDir, setSortDir] = React.useState('desc'); // 'asc' | 'desc'

  const toggleSort = (key) => {
    if (sortKey === key) {
      setSortDir((d) => (d === 'asc' ? 'desc' : 'asc'));
    } else {
      setSortKey(key);
      setSortDir('asc');
    }
  };

  const sortedPortfolio = React.useMemo(() => {
    const copy = [...portfolio];
    copy.sort((a, b) => {
      let av, bv;
      if (sortKey === 'allocation') {
        av = Number(a.allocation) || 0;
        bv = Number(b.allocation) || 0;
      } else if (sortKey === 'company') {
        av = (a.abbr || a.ticker || '').toLowerCase();
        bv = (b.abbr || b.ticker || '').toLowerCase();
      } else if (sortKey === 'sector') {
        av = (a.sector || '').toLowerCase();
        bv = (b.sector || '').toLowerCase();
      } else {
        av = 0; bv = 0;
      }
      if (av < bv) return sortDir === 'asc' ? -1 : 1;
      if (av > bv) return sortDir === 'asc' ? 1 : -1;
      return 0;
    });
    return copy;
  }, [portfolio, sortKey, sortDir]);

  const sectorDistribution = React.useMemo(() => {
    const agg = {};
    portfolio.forEach((row) => {
      const sec = row.sector || 'Unknown';
      const val = Number(row.allocation) || 0;
      agg[sec] = (agg[sec] || 0) + val;
    });
    return Object.entries(agg).map(([name, value]) => ({ name, value }));
  }, [portfolio]);

  // Map sector -> top contributing tickers for tooltip relevance to table
  const sectorContributors = React.useMemo(() => {
    const map = {};
    portfolio.forEach((row) => {
      const sec = row.sector || 'Unknown';
      const val = Number(row.allocation) || 0;
      if (!map[sec]) map[sec] = [];
      map[sec].push({ abbr: row.abbr || row.ticker, allocation: val });
    });
    Object.keys(map).forEach((sec) => {
      map[sec].sort((a, b) => b.allocation - a.allocation);
      map[sec] = map[sec].slice(0, 4);
    });
    return map;
  }, [portfolio]);

  const COLORS = ['#6366f1', '#10b981', '#f59e0b', '#ef4444', '#3b82f6', '#8b5cf6', '#14b8a6', '#f43f5e', '#84cc16', '#a78bfa'];

  // Custom in-slice percentage label to avoid cutting off
  const renderSectorLabel = ({ cx, cy, midAngle, innerRadius, outerRadius, percent }) => {
    if (!isFinite(percent)) return null;
    const RADIAN = Math.PI / 180;
    const r = innerRadius + (outerRadius - innerRadius) * 0.6; // place label ~60% into the slice
    const x = cx + r * Math.cos(-midAngle * RADIAN);
    const y = cy + r * Math.sin(-midAngle * RADIAN);
    const text = `${(percent * 100).toFixed(1)}%`;
    return (
      <text x={x} y={y} fill="#374151" textAnchor="middle" dominantBaseline="central" style={{ fontSize: 12, fontWeight: 600 }}>
        {text}
      </text>
    );
  };

  return (
    <div className="dashboard-container" style={{ maxWidth: '1100px', margin: '0 auto', padding: '2vw', minHeight: '70vh', background: '#f3f4f6' }}>
      <div className="card" style={{ background: '#fff', borderRadius: '1.5rem', boxShadow: '0 2px 12px rgba(30,41,59,0.08)', padding: '2rem' }}>
        <h2 style={{ fontSize: '1.6rem', fontWeight: 700, marginBottom: '0.75rem', color: '#22223b', fontFamily: 'Poppins, Inter, sans-serif', letterSpacing: '0.2px' }}>Portfolio Results</h2>

        {hasApi && (
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem', alignItems: 'start', marginBottom: '1.5rem' }}>
            <div>
              {sectors.length > 0 && (
                <div style={{ marginBottom: '1rem' }}>
                  <div style={{ color: '#6b7280', fontSize: '0.85rem', marginBottom: '0.5rem' }}>Sectors</div>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
                    {sectors.map(s => (
                      <span key={s} style={{ background: '#f3f4f6', color: '#22223b', borderRadius: '999px', padding: '0.25rem 0.6rem', fontSize: '0.85rem', fontWeight: 600 }}>{s}</span>
                    ))}
                  </div>
                </div>
              )}
              <div style={{ marginBottom: '1.25rem' }}>
                <div style={{ color: '#6b7280', fontSize: '0.95rem', marginBottom: '0.5rem' }}>Companies Data Used</div>
                <div style={{ color: '#22223b', fontSize: '1rem', fontWeight: 600 }}>{companiesUsedCount}</div>
              </div>
              {assetsUsed.length > 0 && (
                <div style={{ marginBottom: '1rem' }}>
                  <div style={{ color: '#6b7280', fontSize: '0.85rem', marginBottom: '0.5rem' }}>Commodities Used</div>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
                    {assetsUsed.map(a => (
                      <span key={a} style={{ background: '#eef2ff', color: '#1f2937', borderRadius: '999px', padding: '0.25rem 0.6rem', fontSize: '0.85rem', fontWeight: 600 }}>{a}</span>
                    ))}
                  </div>
                </div>
              )}
            </div>
            {sectorDistribution.length > 0 && (
              <div style={{ minWidth: '320px', height: '340px' }}>
                <div style={{ color: '#6b7280', fontSize: '0.95rem', marginBottom: '0.5rem' }}>Capital Allocation</div>
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={sectorDistribution}
                      dataKey="value"
                      nameKey="name"
                      innerRadius={80}
                      outerRadius={130}
                      padAngle={3}
                      cornerRadius={6}
                      minAngle={8}
                      stroke="#fff"
                      labelLine={false}
                      label={renderSectorLabel}
                    >
                      {sectorDistribution.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip content={({ active, payload }) => {
                      if (!active || !payload || !payload.length) return null;
                      const p = payload[0];
                      const sector = p.name;
                      const val = Number(p.value ?? (p.payload && p.payload.value));
                      const total = sectorDistribution.reduce((sum, d) => sum + (Number(d.value) || 0), 0);
                      const pct = total > 0 && isFinite(val) ? ((val / total) * 100).toFixed(2) : '0.00';
                      const contrib = sectorContributors[sector] || [];
                      return (
                        <div style={{ background: '#fff', border: '1px solid #e5e7eb', borderRadius: 8, boxShadow: '0 2px 10px rgba(30,41,59,0.10)', padding: '0.6rem 0.8rem' }}>
                          <div style={{ fontWeight: 700, color: '#111827', marginBottom: 6 }}>{sector} : {pct}%</div>
                          {contrib.length > 0 && (
                            <div style={{ display: 'grid', gap: 4 }}>
                              {contrib.map((c) => (
                                <div key={`${sector}-${c.abbr}`} style={{ display: 'flex', gap: 6, fontSize: 12, color: '#374151' }}>
                                  <span style={{ fontWeight: 600 }}>{c.abbr}</span>
                                  <span style={{ marginLeft: 'auto' }}>{Number(c.allocation).toFixed(2)}%</span>
                                </div>
                              ))}
                            </div>
                          )}
                        </div>
                      );
                    }} />
                    <Legend verticalAlign="bottom" align="center" />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            )}
          </div>
        )}

        {hasApi && portfolio.length > 0 ? (
          <>
            <table style={{ width: '100%', borderCollapse: 'collapse', marginTop: '0.5rem' }}>
              <thead style={{ fontSize: '0.95rem' }}>
                <tr>
                  <th onClick={() => toggleSort('company')} style={{ cursor: 'pointer', userSelect: 'none', textAlign: 'left', padding: '0.75rem 1rem', color: '#22223b', fontWeight: 600, whiteSpace: 'nowrap' }}>
                    Company
                    <span style={{ marginLeft: 6, color: sortKey === 'company' ? '#111827' : '#9CA3AF' }}>
                      {sortKey === 'company' ? (sortDir === 'asc' ? '▲' : '▼') : '↕'}
                    </span>
                  </th>
                  <th onClick={() => toggleSort('sector')} style={{ cursor: 'pointer', userSelect: 'none', textAlign: 'left', padding: '0.75rem 1rem', color: '#22223b', fontWeight: 600, whiteSpace: 'nowrap' }}>
                    Sector
                    <span style={{ marginLeft: 6, color: sortKey === 'sector' ? '#111827' : '#9CA3AF' }}>
                      {sortKey === 'sector' ? (sortDir === 'asc' ? '▲' : '▼') : '↕'}
                    </span>
                  </th>
                  <th onClick={() => toggleSort('allocation')} style={{ cursor: 'pointer', userSelect: 'none', textAlign: 'right', padding: '0.75rem 1rem', color: '#22223b', fontWeight: 600, whiteSpace: 'nowrap' }}>
                    Allocation
                    <span style={{ marginLeft: 6, color: sortKey === 'allocation' ? '#111827' : '#9CA3AF' }}>
                      {sortKey === 'allocation' ? (sortDir === 'asc' ? '▲' : '▼') : '↕'}
                    </span>
                  </th>
                </tr>
              </thead>
              <tbody style={{ fontSize: '0.95rem' }}>
                {(showAll ? sortedPortfolio : sortedPortfolio.slice(0, 10)).map(row => (
                  <tr key={`${row.ticker}-${row.sector}`}>
                    <td style={{ padding: '0.75rem 1rem', color: '#22223b', fontWeight: 600 }}>{row.abbr || row.ticker}</td>
                    <td style={{ padding: '0.75rem 1rem', color: '#22223b' }}>{row.sector}</td>
                    <td style={{ padding: '0.75rem 1rem', color: '#22223b', textAlign: 'right' }}>{`${Number(row.allocation).toFixed(2)}%`}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {portfolio.length > 10 && (
              <div style={{ display: 'flex', justifyContent: 'center', marginTop: '0.75rem' }}>
                <button
                  onClick={() => setShowAll(v => !v)}
                  style={{
                    background: '#fff',
                    color: '#22223b',
                    border: '1px solid #e5e7eb',
                    borderRadius: '999px',
                    padding: '0.5rem 1rem',
                    fontSize: '0.95rem',
                    fontWeight: 600,
                    cursor: 'pointer'
                  }}
                >{showAll ? 'View less' : 'View more'}</button>
              </div>
            )}
          </>
        ) : (
          <div style={{ marginTop: '0.5rem', color: '#6b7280' }}>No portfolio results available.</div>
        )}
      </div>
    </div>
  );
}

function AssetPerformanceTable({ data, checked, handleCheck, onCreatePortfolio, sp500Averages, subsectors, subChecked, setSubChecked }) {
  const [expanded, setExpanded] = React.useState(false);
  const isSubsectorsProvided = Array.isArray(subsectors) && subsectors.length > 0;
  const isStockMarketChecked = subChecked.every(Boolean);

  // Commodities: fetch and manage expand + selection
  const [commodities, setCommodities] = React.useState([]);
  const [comExpanded, setComExpanded] = React.useState(false);
  const [comSubChecked, setComSubChecked] = React.useState([]);

  React.useEffect(() => {
    let isMounted = true;
    fetch('/model/commodities/returns/')
      .then((res) => res.json())
      .then((json) => {
        if (!isMounted || !Array.isArray(json)) return;
        setCommodities(json);
        setComSubChecked(Array(json.length).fill(false));
      })
      .catch(() => {
        // silently ignore for now
      });
    return () => { isMounted = false; };
  }, []);

  // Select all/deselect all logic for S&P 500
  const handleSP500Check = () => {
    const newVal = !isStockMarketChecked;
    setSubChecked(Array(subsectors.length).fill(newVal));
  };

  const handleSubCheck = (idx) => {
    setSubChecked((prev) => {
      const updated = [...prev];
      updated[idx] = !updated[idx];
      return updated;
    });
  };

  const handleComAllCheck = () => {
    const newVal = comSubChecked.length > 0 ? !comSubChecked.every(Boolean) : true;
    setComSubChecked(Array(commodities.length).fill(newVal));
  };

  const handleComSubCheck = (idx) => {
    setComSubChecked((prev) => {
      const updated = [...prev];
      updated[idx] = !updated[idx];
      return updated;
    });
  };

  // Button enabled if any checked OR any subChecked
  const isAnySelected = checked.some(Boolean) || subChecked.some(Boolean) || comSubChecked.some(Boolean);

  return (
    <>
      <div style={{
        background: '#fff',
        borderRadius: '1.25rem',
        boxShadow: '0 1px 8px rgba(30,41,59,0.06)',
        padding: '1.25rem',
        margin: '0 auto',
        maxWidth: '98%',
        overflowX: 'auto',
      }}>
        <table className="data-table" style={{ width: '100%', borderCollapse: 'separate', borderSpacing: 0, background: 'transparent', borderRadius: '1.25rem', overflow: 'hidden' }}>
          <thead>
            <tr>
              <th style={{ background: '#fff', width: '48px' }}></th>
              <th style={{ background: '#fff', color: '#22223b', fontWeight: 700, fontSize: '1.05rem', padding: '0.75rem 1rem' }}>Asset Class</th>
              <th style={{ background: '#fff', color: '#22223b', fontWeight: 700, fontSize: '1.05rem', padding: '0.75rem 1rem' }}>1Y</th>
              <th style={{ background: '#fff', color: '#22223b', fontWeight: 700, fontSize: '1.05rem', padding: '0.75rem 1rem' }}>3Y</th>
              <th style={{ background: '#fff', color: '#22223b', fontWeight: 700, fontSize: '1.05rem', padding: '0.75rem 1rem' }}>5Y</th>
            </tr>
          </thead>
          <tbody>
            {data.map((row, idx) => (
              <React.Fragment key={row.asset}>
                <tr style={{ borderBottom: '1px solid #f3f4f6' }}>
                  <td style={{ textAlign: 'center' }}>
                    {row.asset === 'US Stock Market (S&P 500)' ? (
                      <input
                        type="checkbox"
                        checked={isStockMarketChecked}
                        onChange={handleSP500Check}
                        style={{ width: '20px', height: '20px', accentColor: '#6366f1', cursor: 'pointer' }}
                      />
                    ) : (
                      <input
                        type="checkbox"
                        checked={checked[idx]}
                        onChange={() => handleCheck(idx)}
                        style={{ width: '20px', height: '20px', accentColor: '#6366f1', cursor: 'pointer' }}
                      />
                    )}
                  </td>
                  <td style={{ color: '#22223b', fontWeight: 600, fontSize: '1rem', padding: '0.85rem 1rem' }}>
                    {row.asset === 'US Stock Market (S&P 500)' ? (
                      <span style={{ position: 'relative' }}>
                        <span
                          style={{ cursor: 'pointer' }}
                          onClick={() => setExpanded((v) => !v)}
                        >
                          US Stock Market (S&P 500) {expanded ? '▲' : '▼'}
                        </span>
                      </span>
                    ) : (
                      row.asset
                    )}
                  </td>
                    {(() => { const vals = (row.asset === 'US Stock Market (S&P 500)' && sp500Averages) ? sp500Averages : { y1: row.y1, y3: row.y3, y5: row.y5 }; return (
                    <>
                      <td style={{ background: vals.y1 > 0 ? '#d1fae5' : vals.y1 < 0 ? '#fee2e2' : '#f3f4f6', color: vals.y1 > 0 ? '#065f46' : vals.y1 < 0 ? '#991b1b' : '#22223b', borderRadius: '0.6rem', padding: '0.4rem 0.75rem', fontWeight: 600 }}>{vals.y1}%</td>
                      <td style={{ background: vals.y3 > 0 ? '#d1fae5' : vals.y3 < 0 ? '#fee2e2' : '#f3f4f6', color: vals.y3 > 0 ? '#065f46' : vals.y3 < 0 ? '#991b1b' : '#22223b', borderRadius: '0.6rem', padding: '0.4rem 0.75rem', fontWeight: 600 }}>{vals.y3}%</td>
                      <td style={{ background: vals.y5 > 0 ? '#d1fae5' : vals.y5 < 0 ? '#fee2e2' : '#f3f4f6', color: vals.y5 > 0 ? '#065f46' : '#991b1b', borderRadius: '0.6rem', padding: '0.4rem 0.75rem', fontWeight: 600 }}>{vals.y5}%</td>
                    </>
                  ); })()}
                </tr>
                  {row.asset === 'US Stock Market (S&P 500)' && expanded && isSubsectorsProvided && (
                    subsectors.map((sub, subIdx) => (
                    <tr key={sub.asset} style={{ background: '#f9fafb' }}>
                      <td style={{ textAlign: 'center' }}>
                        <input
                          type="checkbox"
                          checked={subChecked[subIdx]}
                          onChange={() => handleSubCheck(subIdx)}
                          style={{ width: '18px', height: '18px', accentColor: '#6366f1', marginRight: '0.5rem' }}
                        />
                      </td>
                      <td style={{ color: '#22223b', fontWeight: 500, fontSize: '0.95rem', padding: '0.65rem 1rem', paddingLeft: '2rem' }}>{sub.asset}</td>
                      <td style={{ background: sub.y1 > 0 ? '#d1fae5' : sub.y1 < 0 ? '#fee2e2' : '#f3f4f6', color: sub.y1 > 0 ? '#065f46' : sub.y1 < 0 ? '#991b1b' : '#22223b', borderRadius: '0.6rem', padding: '0.4rem 0.75rem', fontWeight: 600 }}>{sub.y1}%</td>
                      <td style={{ background: sub.y3 > 0 ? '#d1fae5' : sub.y3 < 0 ? '#fee2e2' : '#f3f4f6', color: sub.y3 > 0 ? '#065f46' : sub.y3 < 0 ? '#991b1b' : '#22223b', borderRadius: '0.6rem', padding: '0.4rem 0.75rem', fontWeight: 600 }}>{sub.y3}%</td>
                      <td style={{ background: sub.y5 > 0 ? '#d1fae5' : sub.y5 < 0 ? '#fee2e2' : '#f3f4f6', color: sub.y5 > 0 ? '#065f46' : '#991b1b', borderRadius: '0.6rem', padding: '0.4rem 0.75rem', fontWeight: 600 }}>{sub.y5}%</td>
                    </tr>
                  ))
                )}
              </React.Fragment>
            ))}

            {/* Commodities aggregate row */}
            {commodities.length > 0 && (
              <>
                <tr style={{ borderBottom: '1px solid #f3f4f6' }}>
                  <td style={{ textAlign: 'center' }}>
                    <input
                      type="checkbox"
                      checked={comSubChecked.length > 0 && comSubChecked.every(Boolean)}
                      onChange={handleComAllCheck}
                      style={{ width: '20px', height: '20px', accentColor: '#6366f1', cursor: 'pointer' }}
                    />
                  </td>
                  <td style={{ color: '#22223b', fontWeight: 600, fontSize: '1rem', padding: '0.85rem 1rem' }}>
                    <span style={{ cursor: 'pointer' }} onClick={() => setComExpanded((v) => !v)}>
                      Commodities {comExpanded ? '▲' : '▼'}
                    </span>
                  </td>
                  {(() => {
                    const avg = (key) => {
                      const vals = commodities.map((c) => Number(c[key] || 0));
                      const m = vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : 0;
                      return Math.round(m * 100) / 100;
                    };
                    const y1 = avg('y1');
                    const y3 = avg('y3');
                    const y5 = avg('y5');
                    return (
                      <>
                        <td style={{ background: y1 > 0 ? '#d1fae5' : y1 < 0 ? '#fee2e2' : '#f3f4f6', color: y1 > 0 ? '#065f46' : y1 < 0 ? '#991b1b' : '#22223b', borderRadius: '0.6rem', padding: '0.4rem 0.75rem', fontWeight: 600 }}>{y1}%</td>
                        <td style={{ background: y3 > 0 ? '#d1fae5' : y3 < 0 ? '#fee2e2' : '#f3f4f6', color: y3 > 0 ? '#065f46' : y3 < 0 ? '#991b1b' : '#22223b', borderRadius: '0.6rem', padding: '0.4rem 0.75rem', fontWeight: 600 }}>{y3}%</td>
                        <td style={{ background: y5 > 0 ? '#d1fae5' : y5 < 0 ? '#fee2e2' : '#f3f4f6', color: y5 > 0 ? '#065f46' : '#991b1b', borderRadius: '0.6rem', padding: '0.4rem 0.75rem', fontWeight: 600 }}>{y5}%</td>
                      </>
                    );
                  })()}
                </tr>
                {comExpanded && commodities.map((sub, subIdx) => (
                  <tr key={sub.asset} style={{ background: '#f9fafb' }}>
                    <td style={{ textAlign: 'center' }}>
                      <input
                        type="checkbox"
                        checked={comSubChecked[subIdx]}
                        onChange={() => handleComSubCheck(subIdx)}
                        style={{ width: '18px', height: '18px', accentColor: '#6366f1', marginRight: '0.5rem' }}
                      />
                    </td>
                    <td style={{ color: '#22223b', fontWeight: 500, fontSize: '0.95rem', padding: '0.65rem 1rem', paddingLeft: '2rem' }}>{sub.asset}</td>
                    <td style={{ background: sub.y1 > 0 ? '#d1fae5' : sub.y1 < 0 ? '#fee2e2' : '#f3f4f6', color: sub.y1 > 0 ? '#065f46' : sub.y1 < 0 ? '#991b1b' : '#22223b', borderRadius: '0.6rem', padding: '0.4rem 0.75rem', fontWeight: 600 }}>{sub.y1}%</td>
                    <td style={{ background: sub.y3 > 0 ? '#d1fae5' : sub.y3 < 0 ? '#fee2e2' : '#f3f4f6', color: sub.y3 > 0 ? '#065f46' : sub.y3 < 0 ? '#991b1b' : '#22223b', borderRadius: '0.6rem', padding: '0.4rem 0.75rem', fontWeight: 600 }}>{sub.y3}%</td>
                    <td style={{ background: sub.y5 > 0 ? '#d1fae5' : sub.y5 < 0 ? '#fee2e2' : '#f3f4f6', color: sub.y5 > 0 ? '#065f46' : '#991b1b', borderRadius: '0.6rem', padding: '0.4rem 0.75rem', fontWeight: 600 }}>{sub.y5}%</td>
                  </tr>
                ))}
              </>
            )}
          </tbody>
        </table>
      </div>
      <div style={{ textAlign: 'right', marginTop: '1rem' }}>
        <button
          className="btn-primary"
          disabled={!isAnySelected}
          onClick={isAnySelected ? () => {
            const selectedSectors = subsectors.filter((s, i) => subChecked[i]).map(s => s.asset);
            const selectedCommodities = commodities.filter((c, i) => comSubChecked[i]).map(c => c.asset);
            onCreatePortfolio({ sectors: selectedSectors, commodities: selectedCommodities });
          } : undefined}
          style={{
            background: !isAnySelected ? '#e5e7eb' : '#22223b',
            color: !isAnySelected ? '#888' : '#fff',
            border: 'none',
            borderRadius: '0.9rem',
            padding: '0.6rem 1.4rem',
            fontSize: '0.95rem',
            fontWeight: 600,
            boxShadow: '0 1px 6px rgba(30,41,59,0.08)',
            cursor: !isAnySelected ? 'not-allowed' : 'pointer',
            transition: 'background 0.2s, color 0.2s',
          }}
        >
          Create a Portfolio
        </button>
      </div>
    </>
  );
}

function SP500StocksTable({ subsectors, subChecked, handleSubCheck }) {
  return (
    <div style={{
      background: '#fff',
      borderRadius: '1.5rem',
      boxShadow: '0 2px 12px rgba(30,41,59,0.08)',
      padding: '2rem',
      margin: '0 auto',
      maxWidth: '98%',
      overflowX: 'auto',
    }}>
      <table className="data-table" style={{ width: '100%', borderCollapse: 'separate', borderSpacing: 0, background: 'transparent', borderRadius: '1.25rem', overflow: 'hidden' }}>
        <thead>
          <tr>
            <th style={{ background: '#fff', width: '48px' }}></th>
            <th style={{ background: '#fff', color: '#22223b', fontWeight: 700, fontSize: '1.3rem', padding: '1rem 1.5rem' }}>Sectors</th>
            <th style={{ background: '#fff', color: '#22223b', fontWeight: 700, fontSize: '1.3rem', padding: '1rem 1.5rem' }}>1Y</th>
            <th style={{ background: '#fff', color: '#22223b', fontWeight: 700, fontSize: '1.3rem', padding: '1rem 1.5rem' }}>3Y</th>
            <th style={{ background: '#fff', color: '#22223b', fontWeight: 700, fontSize: '1.3rem', padding: '1rem 1.5rem' }}>5Y</th>
          </tr>
        </thead>
        <tbody>
          {subsectors.map((sub, subIdx) => (
            <tr key={sub.asset} style={{ background: '#f9fafb' }}>
              <td style={{ textAlign: 'center' }}>
                <input
                  type="checkbox"
                  checked={subChecked[subIdx]}
                  onChange={() => handleSubCheck(subIdx)}
                  style={{ width: '18px', height: '18px', accentColor: '#6366f1', marginRight: '0.5rem' }}
                />
              </td>
              <td style={{ color: '#22223b', fontWeight: 400, fontSize: '1rem', padding: '0.75rem 1.5rem' }}>{sub.asset}</td>
              <td style={{ background: sub.y1 > 0 ? '#d1fae5' : sub.y1 < 0 ? '#fee2e2' : '#f3f4f6', color: sub.y1 > 0 ? '#065f46' : sub.y1 < 0 ? '#991b1b' : '#22223b', borderRadius: '0.75rem', padding: '0.5rem 1rem', fontWeight: 500 }}>{sub.y1}%</td>
              <td style={{ background: sub.y3 > 0 ? '#d1fae5' : sub.y3 < 0 ? '#fee2e2' : '#f3f4f6', color: sub.y3 > 0 ? '#065f46' : sub.y3 < 0 ? '#991b1b' : '#22223b', borderRadius: '0.75rem', padding: '0.5rem 1rem', fontWeight: 500 }}>{sub.y3}%</td>
              <td style={{ background: sub.y5 > 0 ? '#d1fae5' : sub.y5 < 0 ? '#fee2e2' : '#f3f4f6', color: sub.y5 > 0 ? '#065f46' : sub.y5 < 0 ? '#991b1b' : '#22223b', borderRadius: '0.75rem', padding: '0.5rem 1rem', fontWeight: 500 }}>{sub.y5}%</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function TemplatesPage() {
  return (
    <div className="dashboard-container" style={{ maxWidth: '1400px', margin: '0 auto', padding: '2vw', minHeight: '70vh', background: '#f3f4f6' }}>
      <div className="card" style={{ background: '#fff', borderRadius: '1.5rem', boxShadow: '0 2px 12px rgba(30,41,59,0.08)', padding: '2rem', margin: '0 auto', maxWidth: '900px' }}>
        <h2 style={{ fontSize: '2.2rem', fontWeight: 700, marginBottom: '1.5rem', color: '#22223b', fontFamily: 'Poppins, Inter, sans-serif' }}>Composite</h2>
        <p style={{ color: '#22223b', fontSize: '1.1rem', fontFamily: 'Poppins, Inter, sans-serif' }}>
          Here you can manage and select portfolio composite. (Demo content)
        </p>
        <ul style={{ marginTop: '2rem', paddingLeft: 0, listStyle: 'none' }}>
          <li style={{ background: '#f3f4f6', borderRadius: '1rem', padding: '1rem 1.5rem', marginBottom: '1rem', color: '#22223b', fontFamily: 'Poppins, Inter, sans-serif', fontWeight: 500 }}>Conservative Portfolio</li>
          <li style={{ background: '#f3f4f6', borderRadius: '1rem', padding: '1rem 1.5rem', marginBottom: '1rem', color: '#22223b', fontFamily: 'Poppins, Inter, sans-serif', fontWeight: 500 }}>Balanced Portfolio</li>
          <li style={{ background: '#f3f4f6', borderRadius: '1rem', padding: '1rem 1.5rem', marginBottom: '1rem', color: '#22223b', fontFamily: 'Poppins, Inter, sans-serif', fontWeight: 500 }}>Aggressive Portfolio</li>
        </ul>
      </div>
    </div>
  );
}

function DashboardDemo() {

  const [activeSegment, setActiveSegment] = React.useState(0);
  const segments = ['Asset Classes', 'S&P 500 Stocks', 'Composite', 'Portfolios'];
  const [checked, setChecked] = React.useState(Array(assetPerformanceData.length).fill(false));
  const [showPortfolio, setShowPortfolio] = React.useState(false);
  const [loading, setLoading] = React.useState(false);
  const [portfolioData, setPortfolioData] = React.useState(null);
  const [sectorReturns, setSectorReturns] = React.useState(null);
  const [sp500Averages, setSp500Averages] = React.useState(null);
  const [commoditiesReturns, setCommoditiesReturns] = React.useState(null);
  const [commoditiesAverages, setCommoditiesAverages] = React.useState(null);
  const [sectorSeries, setSectorSeries] = React.useState(null);
  const [hoveredSeries, setHoveredSeries] = React.useState(null);
  const [lockedSeries, setLockedSeries] = React.useState(null);
  const [range, setRange] = React.useState('Max');
  const chartData = React.useMemo(() => {
    if (!sectorSeries || Object.keys(sectorSeries).length === 0) return [];
    const sectors = Object.keys(sectorSeries);
    const dateSet = new Set();
    sectors.forEach(sec => {
      sectorSeries[sec].forEach(pt => dateSet.add(pt.date));
    });
    const dates = Array.from(dateSet).sort();
    const merged = dates.map(d => {
      const row = { date: d };
      sectors.forEach(sec => {
        const arr = sectorSeries[sec];
        const found = arr.find(pt => pt.date === d);
        row[sec] = found ? found.value : null;
      });
      return row;
    });
    if (!merged.length) return merged;
    const end = new Date(merged[merged.length - 1].date);
    let start;
    switch (range) {
      case '1D':
        start = new Date(end); start.setDate(start.getDate() - 1); break;
      case '5D':
        start = new Date(end); start.setDate(start.getDate() - 5); break;
      case '1M':
        start = new Date(end); start.setMonth(start.getMonth() - 1); break;
      case '6M':
        start = new Date(end); start.setMonth(start.getMonth() - 6); break;
      case '1Y':
        start = new Date(end); start.setFullYear(start.getFullYear() - 1); break;
      case '5Y':
        start = new Date(end); start.setFullYear(start.getFullYear() - 5); break;
      case 'Max':
      default:
        start = new Date(merged[0].date);
        break;
    }
    return merged.filter(r => {
      const dt = new Date(r.date);
      return dt >= start && dt <= end;
    });
  }, [sectorSeries, range]);

  // Y-axis ticks at fixed $50 intervals across visible range
  const yTicks = React.useMemo(() => {
    if (!chartData || chartData.length === 0) return [];
    let minV = Infinity;
    let maxV = -Infinity;
    const keys = Object.keys(sectorSeries || {});
    chartData.forEach(row => {
      keys.forEach(k => {
        const v = row[k];
        if (typeof v === 'number' && !Number.isNaN(v)) {
          if (v < minV) minV = v;
          if (v > maxV) maxV = v;
        }
      });
    });
    if (!isFinite(minV) || !isFinite(maxV)) return [];
    const start = Math.floor(minV / 50) * 50;
    const end = Math.ceil(maxV / 50) * 50;
    const ticks = [];
    for (let t = start; t <= end; t += 50) ticks.push(t);
    return ticks;
  }, [chartData, sectorSeries]);

  // Configure API base to work in dev (CRA) and production
  const API_BASE = process.env.REACT_APP_API_BASE || '';

  React.useEffect(() => {
    const controller = new AbortController();
    console.log('[DashboardDemo] Fetching sector returns from', `${API_BASE}/model/sp500/gics-returns/`);
    fetch(`${API_BASE}/model/sp500/gics-returns/`, { signal: controller.signal })
      .then(res => res.ok ? res.json() : Promise.reject(new Error('Failed to fetch sector returns')))
      .then(data => {
        console.log('[DashboardDemo] Sector returns response:', data);
        if (Array.isArray(data)) {
          setSectorReturns(data);
          const sum = data.reduce((acc, s) => ({
            y1: acc.y1 + (Number(s.y1) || 0),
            y3: acc.y3 + (Number(s.y3) || 0),
            y5: acc.y5 + (Number(s.y5) || 0),
          }), { y1: 0, y3: 0, y5: 0 });
          const n = data.length || 1;
          setSp500Averages({
            y1: +(sum.y1 / n).toFixed(2),
            y3: +(sum.y3 / n).toFixed(2),
            y5: +(sum.y5 / n).toFixed(2),
          });
        }
      })
      .catch((err) => {
        console.error('[DashboardDemo] Fetch sector returns error:', err);
      });
    return () => controller.abort();
  }, []);

  // Fetch Commodities returns (1Y/3Y/5Y) from backend if available
  React.useEffect(() => {
    const controller = new AbortController();
    fetch(`${API_BASE}/model/commodities/returns/`, { signal: controller.signal })
      .then(res => res.ok ? res.json() : Promise.reject(new Error('Failed to fetch commodities returns')))
      .then(data => {
        if (Array.isArray(data)) {
          setCommoditiesReturns(data);
          const sum = data.reduce((acc, s) => ({
            y1: acc.y1 + (Number(s.y1) || 0),
            y3: acc.y3 + (Number(s.y3) || 0),
            y5: acc.y5 + (Number(s.y5) || 0),
          }), { y1: 0, y3: 0, y5: 0 });
          const n = data.length || 1;
          setCommoditiesAverages({
            y1: +(sum.y1 / n).toFixed(2),
            y3: +(sum.y3 / n).toFixed(2),
            y5: +(sum.y5 / n).toFixed(2),
          });
        }
      })
      .catch(() => {})
    return () => controller.abort();
  }, [API_BASE]);

  // Fetch monthly sector time series for chart (downsampled by backend)
  React.useEffect(() => {
    const controller = new AbortController();
    console.log('[DashboardDemo] Fetching sector timeseries from', `${API_BASE}/model/sp500/gics-timeseries/?limit=all`);
    fetch(`${API_BASE}/model/sp500/gics-timeseries/?limit=all`, { signal: controller.signal })
      .then(res => res.ok ? res.json() : Promise.reject(new Error('Failed to fetch sector timeseries')))
      .then(data => {
        console.log('[DashboardDemo] Sector timeseries response:', data);
        if (data && data.series) {
          setSectorSeries(data.series);
        }
      })
      .catch((err) => {
        console.error('[DashboardDemo] Fetch sector timeseries error:', err);
      });
    return () => controller.abort();
  }, [API_BASE]);

  // S&P 500 subsectors derived from backend sector returns
  const subsectors = React.useMemo(() => {
    if (!Array.isArray(sectorReturns) || sectorReturns.length === 0) return [];
    return sectorReturns.map(s => ({
      asset: s.sector,
      y1: Number(s.y1),
      y3: Number(s.y3),
      y5: Number(s.y5),
    }));
  }, [sectorReturns]);
  const [subChecked, setSubChecked] = React.useState([]);
  React.useEffect(() => {
    setSubChecked(Array(subsectors.length).fill(true));
  }, [subsectors.length]);
  const isStockMarketChecked = subChecked.every(Boolean);

  const commoditiesSubsectors = React.useMemo(() => {
    if (!Array.isArray(commoditiesReturns) || commoditiesReturns.length === 0) return [];
    return commoditiesReturns.map(s => ({
      asset: s.asset || s.commodity,
      y1: Number(s.y1),
      y3: Number(s.y3),
      y5: Number(s.y5),
    }));
  }, [commoditiesReturns]);
  const [comSubChecked, setComSubChecked] = React.useState([]);
  React.useEffect(() => {
    setComSubChecked(Array(commoditiesSubsectors.length).fill(true));
  }, [commoditiesSubsectors.length]);

  // Selection logic
  const handleCheck = (idx) => {
    if (assetPerformanceData[idx].asset === 'US Stock Market (S&P 500)') {
      const newVal = !isStockMarketChecked;
      setSubChecked(Array(subsectors.length).fill(newVal));
    } else {
      setChecked((prev) => {
        const updated = [...prev];
        updated[idx] = !updated[idx];
        return updated;
      });
    }
  };

  const handleSubCheck = (idx) => {
    setSubChecked((prev) => {
      const updated = [...prev];
      updated[idx] = !updated[idx];
      return updated;
    });
  };

  const handleCreatePortfolio = async ({ sectors, commodities }) => {
    try {
      setLoading(true);
      const res = await fetch(`${API_BASE}/model/train_by_sectors/`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sectors: Array.isArray(sectors) ? sectors : [],
          Commodities: Array.isArray(commodities) ? commodities : [],
          top_k: 'all',
        }),
      });
      if (!res.ok) throw new Error('Training request failed');
      const result = await res.json();

      setPortfolioData({ result });
      setShowPortfolio(true);
      setActiveSegment(3);
    } catch (err) {
      console.error('CreatePortfolio error:', err);
    } finally {
      setLoading(false);
    }
  };

  const getRowVals = (row) => {
    if (row.asset === 'US Stock Market (S&P 500)' && sp500Averages) {
      return sp500Averages;
    }
    return { y1: row.y1, y3: row.y3, y5: row.y5 };
  };

  return (
    <div className="dashboard-container" style={{ maxWidth: '1400px', margin: '0 auto', padding: '2vw' }}>
      <DashboardHeader />
      <SegmentedControl segments={segments} active={activeSegment} onChange={setActiveSegment} />
      {/* Loading overlay */}
      {loading && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          width: '100vw',
          height: '100vh',
          background: 'rgba(34,34,59,0.15)',
          zIndex: 9999,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        }}>
          <div style={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            background: 'rgba(255,255,255,0.85)',
            borderRadius: '2rem',
            boxShadow: '0 2px 24px rgba(30,41,59,0.12)',
            padding: '2.5rem 3rem',
          }}>
            <div style={{ marginBottom: '1.5rem' }}>
              <svg width="64" height="64" viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg" style={{ animation: 'spin 1.2s linear infinite' }}>
                <circle cx="32" cy="32" r="28" stroke="#6366f1" strokeWidth="8" strokeDasharray="44 44" strokeDashoffset="0" />
                <style>{`@keyframes spin { 100% { transform: rotate(360deg); } }`}</style>
              </svg>
            </div>
            <div style={{ fontSize: '1.3rem', color: '#22223b', fontWeight: 600, fontFamily: 'Poppins, Inter, sans-serif' }}>
              Training model and generating portfolio weights...
            </div>
          </div>
        </div>
      )}
      {/* ...existing code... */}
      {activeSegment === 3 ? (
        !loading ? (
          <PortfolioResults result={portfolioData?.result} />
        ) : null
      ) : activeSegment === 0 ? (
        <div className="card" style={{ display: 'flex', flexWrap: 'wrap', gap: '1.5rem', background: '#f3f4f6', borderRadius: '1.5rem', boxShadow: '0 1px 8px rgba(30,41,59,0.06)', minHeight: '50vh', alignItems: 'stretch' }}>
          <div style={{ flex: '1 1 0', minWidth: '320px', maxWidth: '100%', minHeight: '260px', display: 'flex', alignItems: 'center', justifyContent: 'center', width: '50%', boxSizing: 'border-box' }}>
            <div style={{ width: '100%', height: '100%', background: '#fff', borderRadius: '1.25rem', padding: '1rem' }}>
              <h3 style={{ margin: 0, marginBottom: '0.5rem', color: '#22223b' }}>Asset Class Performance </h3>
              <div style={{ display: 'flex', gap: 10, marginBottom: 8, alignItems: 'center', flexWrap: 'wrap' }}>
                {['1D','5D','1M','6M','1Y','5Y','Max'].map(r => (
                  <button
                    key={r}
                    onClick={() => setRange(r)}
                    style={{
                      padding: '6px 10px',
                      borderRadius: 16,
                      border: '1px solid #e5e7eb',
                      background: range === r ? '#22223b' : '#fff',
                      color: range === r ? '#fff' : '#22223b',
                      fontSize: 12,
                      cursor: 'pointer'
                    }}
                  >{r}</button>
                ))}
                <button
                  onClick={() => { setLockedSeries(null); setHoveredSeries(null); }}
                  style={{
                    marginLeft: 'auto',
                    padding: '6px 10px',
                    borderRadius: 16,
                    border: '1px solid #e5e7eb',
                    background: '#fff',
                    color: '#22223b',
                    fontSize: 12,
                    cursor: 'pointer'
                  }}
                >Reset highlight</button>
              </div>
              <ResponsiveContainer width="100%" height={340}>
                <LineChart data={chartData} margin={{ top: 10, right: 20, left: 10, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" hide={false} tick={{ fontSize: 12 }} interval="preserveStartEnd" tickFormatter={(d) => {
                    try {
                      const dt = new Date(d);
                      if (['1D','5D'].includes(range)) {
                        return `${dt.getHours()}:${String(dt.getMinutes()).padStart(2,'0')}`;
                      }
                      if (['1M','6M','1Y'].includes(range)) {
                        return `${dt.getFullYear()}-${String(dt.getMonth()+1).padStart(2,'0')}`;
                      }
                      return dt.getFullYear();
                    } catch { return d; }
                  }} />
                  <YAxis 
                    domain={[yTicks.length ? yTicks[0] : 'auto', yTicks.length ? yTicks[yTicks.length - 1] : 'auto']}
                    tick={{ fontSize: 13 }}
                    ticks={yTicks}
                    padding={{ top: 8, bottom: 8 }}
                    tickFormatter={(v) => `$${Math.round(v)}`}
                    label={{ value: 'Index (base=100)', angle: -90, position: 'insideLeft', style: { fill: '#6b7280', fontSize: 12 } }}
                  />
                  <Tooltip content={<CompactTooltip />} />
                  {false && <Legend content={<HoverLegend onHover={(key) => setHoveredSeries(key)} onLeave={() => setHoveredSeries(null)} />} />}
                  {(() => {
                    if (!chartData.length) return null;
                    const last = chartData[chartData.length - 1];
                    const baselineY = lockedSeries && typeof last[lockedSeries] === 'number' ? last[lockedSeries] : null;
                    return baselineY ? (
                      <ReferenceLine y={baselineY} stroke="#9ca3af" strokeDasharray="4 4" ifOverflow="extendDomain" label={{ value: 'Previous close', position: 'right', fill: '#6b7280', fontSize: 11 }} />
                    ) : null;
                  })()}
                  {sectorSeries && Object.keys(sectorSeries).map((sec, idx) => (
                    <Line
                      key={sec}
                      type="monotone"
                      dataKey={sec}
                      stroke={["#6366f1","#10b981","#f59e0b","#ef4444","#3b82f6","#8b5cf6","#14b8a6","#f43f5e","#84cc16","#a78bfa"][idx % 10]}
                      dot={false}
                      strokeWidth={(lockedSeries ? lockedSeries === sec : hoveredSeries === sec) ? 2.4 : 1.3}
                      strokeOpacity={(lockedSeries ? lockedSeries !== sec : (hoveredSeries && hoveredSeries !== sec)) ? 0.22 : 1}
                    />
                  ))}
                </LineChart>
              </ResponsiveContainer>
              {sectorSeries && (
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6, marginTop: 10 }}>
                  {Object.keys(sectorSeries).map((sec, idx) => {
                    const color = ["#6366f1","#10b981","#f59e0b","#ef4444","#3b82f6","#8b5cf6","#14b8a6","#f43f5e","#84cc16","#a78bfa"][idx % 10];
                    const isActive = lockedSeries ? lockedSeries === sec : hoveredSeries === sec;
                    const isDim = lockedSeries ? lockedSeries !== sec : (hoveredSeries && hoveredSeries !== sec);
                    return (
                      <div
                        key={sec}
                        style={{ display: 'flex', alignItems: 'center', gap: 6, cursor: 'pointer', opacity: isDim ? 0.6 : 1 }}
                        onMouseEnter={() => !lockedSeries && setHoveredSeries(sec)}
                        onMouseLeave={() => !lockedSeries && setHoveredSeries(null)}
                        onClick={() => setLockedSeries(prev => prev === sec ? null : sec)}
                        title={lockedSeries === sec ? 'Click to unlock' : 'Click to lock highlight'}
                      >
                        <span style={{ width: 9, height: 9, borderRadius: '50%', background: color, display: 'inline-block', boxShadow: isActive ? '0 0 0 2px rgba(99,102,241,0.25)' : 'none' }}></span>
                        <span style={{ color: '#374151', fontSize: 12, fontWeight: isActive ? 600 : 500 }}>{sec}</span>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          </div>
          <div style={{ flex: '1 1 0', minWidth: '300px', maxWidth: '100%', width: '50%', boxSizing: 'border-box', overflowX: 'visible' }}>
            {/* <h2 style={{ fontSize: '1.1rem', fontWeight: 700, marginBottom: '0.75rem', color: '#22223b', letterSpacing: '0.2px' }}>
              Asset Class Performance
            </h2> */}
            <AssetPerformanceTable
              data={assetPerformanceData}
              checked={checked}
              handleCheck={handleCheck}
              onCreatePortfolio={handleCreatePortfolio}
              subsectors={subsectors}
              subChecked={subChecked}
              setSubChecked={setSubChecked}
              commoditiesSubsectors={commoditiesSubsectors}
              comSubChecked={comSubChecked}
              setComSubChecked={setComSubChecked}
              commoditiesAverages={commoditiesAverages}
            />
          </div>
        </div>
      ) : activeSegment === 1 ? (
        // S&P 500 Stocks tab content (reuse your S&P 500 table and chart layout)
        <div className="card" style={{ display: 'flex', flexWrap: 'wrap', gap: '1.5rem', background: '#f3f4f6', borderRadius: '1.5rem', boxShadow: '0 1px 8px rgba(30,41,59,0.06)', minHeight: '50vh', alignItems: 'stretch', padding: '1.25rem' }}>
          <div style={{ flex: '1 1 0', minWidth: '320px', maxWidth: '100%', minHeight: '260px', display: 'flex', alignItems: 'center', justifyContent: 'center', width: '50%', boxSizing: 'border-box' }}>
            <div style={{ width: '100%', height: '100%', background: '#fff', borderRadius: '1.25rem', padding: '1rem' }}>
              <h3 style={{ margin: 0, marginBottom: '0.5rem', color: '#22223b' }}>S&P 500</h3>
              <div style={{ display: 'flex', gap: 10, marginBottom: 8, alignItems: 'center', flexWrap: 'wrap' }}>
                {['1D','5D','1M','6M','1Y','5Y','Max'].map(r => (
                  <button
                    key={r}
                    onClick={() => setRange(r)}
                    style={{
                      padding: '6px 10px',
                      borderRadius: 16,
                      border: '1px solid #e5e7eb',
                      background: range === r ? '#22223b' : '#fff',
                      color: range === r ? '#fff' : '#22223b',
                      fontSize: 12,
                      cursor: 'pointer'
                    }}
                  >{r}</button>
                ))}
                <button
                  onClick={() => { setLockedSeries(null); setHoveredSeries(null); }}
                  style={{
                    marginLeft: 'auto',
                    padding: '6px 10px',
                    borderRadius: 16,
                    border: '1px solid #e5e7eb',
                    background: '#fff',
                    color: '#22223b',
                    fontSize: 12,
                    cursor: 'pointer'
                  }}
                >Reset highlight</button>
              </div>
              <ResponsiveContainer width="100%" height={340}>
                <LineChart data={chartData} margin={{ top: 10, right: 20, left: 10, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" hide={false} tick={{ fontSize: 12 }} interval="preserveStartEnd" tickFormatter={(d) => {
                    try { return new Date(d).getFullYear(); } catch { return d; }
                  }} />
                  <YAxis 
                    domain={[yTicks.length ? yTicks[0] : 'auto', yTicks.length ? yTicks[yTicks.length - 1] : 'auto']}
                    tick={{ fontSize: 13 }}
                    ticks={yTicks}
                    padding={{ top: 8, bottom: 8 }}
                    tickFormatter={(v) => `$${Math.round(v)}`}
                    label={{ value: 'Index (base=100)', angle: -90, position: 'insideLeft', style: { fill: '#6b7280', fontSize: 12 } }}
                  />
                  <Tooltip content={<CompactTooltip />} />
                  {false && <Legend content={<HoverLegend onHover={(key) => setHoveredSeries(key)} onLeave={() => setHoveredSeries(null)} />} />}
                  {(() => {
                    if (!chartData.length) return null;
                    const last = chartData[chartData.length - 1];
                    const baselineY = lockedSeries && typeof last[lockedSeries] === 'number' ? last[lockedSeries] : null;
                    return baselineY ? (
                      <ReferenceLine y={baselineY} stroke="#9ca3af" strokeDasharray="4 4" ifOverflow="extendDomain" label={{ value: 'Previous close', position: 'right', fill: '#6b7280', fontSize: 11 }} />
                    ) : null;
                  })()}
                  {sectorSeries && Object.keys(sectorSeries).map((sec, idx) => (
                    <Line
                      key={sec}
                      type="monotone"
                      dataKey={sec}
                      stroke={["#6366f1","#10b981","#f59e0b","#ef4444","#3b82f6","#8b5cf6","#14b8a6","#f43f5e","#84cc16","#a78bfa"][idx % 10]}
                      dot={false}
                      strokeWidth={(lockedSeries ? lockedSeries === sec : hoveredSeries === sec) ? 2.4 : 1.3}
                      strokeOpacity={(lockedSeries ? lockedSeries !== sec : (hoveredSeries && hoveredSeries !== sec)) ? 0.22 : 1}
                    />
                  ))}
                </LineChart>
              </ResponsiveContainer>
              {sectorSeries && (
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6, marginTop: 10 }}>
                  {Object.keys(sectorSeries).map((sec, idx) => {
                    const color = ["#6366f1","#10b981","#f59e0b","#ef4444","#3b82f6","#8b5cf6","#14b8a6","#f43f5e","#84cc16","#a78bfa"][idx % 10];
                    const isActive = lockedSeries ? lockedSeries === sec : hoveredSeries === sec;
                    const isDim = lockedSeries ? lockedSeries !== sec : (hoveredSeries && hoveredSeries !== sec);
                    return (
                      <div
                        key={sec}
                        style={{ display: 'flex', alignItems: 'center', gap: 6, cursor: 'pointer', opacity: isDim ? 0.6 : 1 }}
                        onMouseEnter={() => !lockedSeries && setHoveredSeries(sec)}
                        onMouseLeave={() => !lockedSeries && setHoveredSeries(null)}
                        onClick={() => setLockedSeries(prev => prev === sec ? null : sec)}
                        title={lockedSeries === sec ? 'Click to unlock' : 'Click to lock highlight'}
                      >
                        <span style={{ width: 9, height: 9, borderRadius: '50%', background: color, display: 'inline-block', boxShadow: isActive ? '0 0 0 2px rgba(99,102,241,0.25)' : 'none' }}></span>
                        <span style={{ color: '#374151', fontSize: 12, fontWeight: isActive ? 600 : 500 }}>{sec}</span>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          </div>
          <div style={{ flex: '1 1 0', minWidth: '300px', maxWidth: '100%', width: '50%', boxSizing: 'border-box', overflowX: 'visible' }}>
            {/* <h2 style={{ fontSize: '1.6rem', fontWeight: 700, marginBottom: '0.75rem', color: '#22223b', letterSpacing: '0.2px' }}>
              S&P 500 Sectors
            </h2> */}
            <SP500StocksTable subsectors={subsectors} subChecked={subChecked} handleSubCheck={handleSubCheck} />
          </div>
        </div>
      ) : (
        <TemplatesPage />
      )}
      <style>{`
        @media (max-width: 1100px) {
          .dashboard-container {
            padding: 1vw !important;
          }
          .card {
            flex-direction: column !important;
            gap: 1rem !important;
            padding: 1rem !important;
          }
        }
        @media (max-width: 600px) {
          .dashboard-title {
            font-size: 1rem !important;
          }
          .card {
            padding: 0.5rem !important;
            border-radius: 1rem !important;
          }
          .data-table th, .data-table td {
            padding: 0.4rem 0.5rem !important;
            font-size: 0.85rem !important;
          }
        }
      `}</style>
    </div>
  );
}

export default DashboardDemo;