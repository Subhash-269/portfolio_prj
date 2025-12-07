import React from 'react';

const assetPerformanceData = [
  { asset: 'Gold', y1: 57.94, y3: 33.02, y5: 18.40 },
  // { asset: 'Intl Developed Markets', y1: 26.26, y3: 15.87, y5: 9.49 },
  // { asset: 'Emerging Markets', y1: 22.81, y3: 14.30, y5: 6.22 },
  { asset: 'US Stock Market (S&P 500)', y1: 17.53, y3: 24.89, y5: 16.45 },
  { asset: 'Commodities', y1: 15.84, y3: 4.26, y5: 11.62 },
  // { asset: 'Intermediate Treasuries', y1: 6.44, y3: 4.17, y5: -0.18 },
  // { asset: 'Total Bond Market', y1: 5.70, y3: 4.52, y5: -0.34 },
  // { asset: 'Short Treasuries', y1: 5.09, y3: 4.44, y5: 1.68 },
  { asset: 'US Bonds', y1: 3.74, y3: 1.96, y5: -4.43 },
  // { asset: 'Long Treasuries', y1: 1.71, y3: 0.59, y5: -7.15 },
];

function DashboardHeader() {
  return (
    <header className="dashboard-header">
      <div className="dashboard-title" style={{ color: '#22223b', fontWeight: 700, fontSize: '2rem' }}>
        AI Portfolio
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
    <div className="segmented-control" style={{ display: 'flex', gap: '1.5rem', marginBottom: '2rem', justifyContent: 'center' }}>
      {segments.map((seg, idx) => (
        <button
          key={seg}
          className={`segment-btn${active === idx ? ' active' : ''}`}
          style={{
            background: active === idx ? '#22223b' : '#f3f4f6',
            color: active === idx ? '#fff' : '#22223b',
            border: 'none',
            borderRadius: '2rem',
            padding: '0.75rem 2rem',
            fontSize: '1.1rem',
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
  const portfolio = hasApi
    ? [...result.portfolio].sort((a, b) => b.allocation - a.allocation)
    : [
        { ticker: 'Gold', allocation: 18, sector: 'Commodities' },
        { ticker: 'S&P 500', allocation: 32, sector: 'Equity' },
        { ticker: 'Commodities', allocation: 12, sector: 'Commodities' },
        { ticker: 'US Bonds', allocation: 38, sector: 'Fixed Income' },
      ];

  return (
    <div className="dashboard-container" style={{ maxWidth: '1100px', margin: '0 auto', padding: '2vw', minHeight: '70vh', background: '#f3f4f6' }}>
      <div className="card" style={{ background: '#fff', borderRadius: '1.5rem', boxShadow: '0 2px 12px rgba(30,41,59,0.08)', padding: '2rem' }}>
        <h2 style={{ fontSize: '2.2rem', fontWeight: 700, marginBottom: '1rem', color: '#22223b', fontFamily: 'Poppins, Inter, sans-serif' }}>Portfolio Results</h2>

        {hasApi && sectors.length > 0 && (
          <div style={{ marginBottom: '1rem' }}>
            <div style={{ color: '#6b7280', fontSize: '0.95rem', marginBottom: '0.5rem' }}>Sectors</div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
              {sectors.map(s => (
                <span key={s} style={{ background: '#f3f4f6', color: '#22223b', borderRadius: '999px', padding: '0.35rem 0.75rem', fontSize: '0.9rem', fontWeight: 600 }}>{s}</span>
              ))}
            </div>
          </div>
        )}

        {hasApi && tickersUsed.length > 0 && (
          <div style={{ marginBottom: '1.25rem' }}>
            <div style={{ color: '#6b7280', fontSize: '0.95rem', marginBottom: '0.5rem' }}>Tickers Used</div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
              {tickersUsed.map(t => (
                <span key={t} style={{ background: '#eef2ff', color: '#3730a3', borderRadius: '0.6rem', padding: '0.35rem 0.6rem', fontSize: '0.9rem', fontWeight: 600 }}>{t}</span>
              ))}
            </div>
          </div>
        )}

        <table style={{ width: '100%', borderCollapse: 'collapse', marginTop: '0.5rem' }}>
          <thead>
            <tr>
              <th style={{ textAlign: 'left', padding: '0.75rem 1rem', color: '#22223b', fontWeight: 600 }}>Ticker</th>
              <th style={{ textAlign: 'left', padding: '0.75rem 1rem', color: '#22223b', fontWeight: 600 }}>Sector</th>
              <th style={{ textAlign: 'right', padding: '0.75rem 1rem', color: '#22223b', fontWeight: 600 }}>Allocation</th>
            </tr>
          </thead>
          <tbody>
            {portfolio.map(row => (
              <tr key={`${row.ticker}-${row.sector}`}>
                <td style={{ padding: '0.75rem 1rem', color: '#22223b', fontWeight: 600 }}>{row.ticker}</td>
                <td style={{ padding: '0.75rem 1rem', color: '#22223b' }}>{row.sector}</td>
                <td style={{ padding: '0.75rem 1rem', color: '#22223b', textAlign: 'right' }}>{`${Number(row.allocation).toFixed(2)}%`}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function AssetPerformanceTable({ data, checked, handleCheck, onCreatePortfolio }) {
  const [expanded, setExpanded] = React.useState(false);
  const subsectors = [
    { asset: 'Communication Services', y1: 8.2, y3: 5.1, y5: 2.3 },
    { asset: 'Consumer Discretionary', y1: 7.5, y3: 6.2, y5: 3.1 },
    { asset: 'Consumer Staples', y1: 6.8, y3: 4.9, y5: 2.7 },
    { asset: 'Energy', y1: 5.9, y3: 3.8, y5: 1.9 },
    { asset: 'Financials', y1: 9.1, y3: 7.2, y5: 4.5 },
    { asset: 'Health Care', y1: 10.3, y3: 8.7, y5: 5.6 },
    { asset: 'Industrials', y1: 6.2, y3: 5.0, y5: 2.8 },
    { asset: 'Information Technology', y1: 12.4, y3: 10.1, y5: 7.3 },
    { asset: 'Materials', y1: 4.7, y3: 3.2, y5: 1.5 },
    { asset: 'Real Estate', y1: 3.5, y3: 2.1, y5: 0.9 },
    { asset: 'Utilities', y1: 2.8, y3: 1.7, y5: 0.6 },
  ];
  const [subChecked, setSubChecked] = React.useState(Array(subsectors.length).fill(true));
  const isStockMarketChecked = subChecked.every(Boolean);

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

  // Button enabled if any checked OR any subChecked
  const isAnySelected = checked.some(Boolean) || subChecked.some(Boolean);

  return (
    <>
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
              <th style={{ background: '#fff', color: '#22223b', fontWeight: 700, fontSize: '1.3rem', padding: '1rem 1.5rem' }}>Asset Class</th>
              <th style={{ background: '#fff', color: '#22223b', fontWeight: 700, fontSize: '1.3rem', padding: '1rem 1.5rem' }}>1Y</th>
              <th style={{ background: '#fff', color: '#22223b', fontWeight: 700, fontSize: '1.3rem', padding: '1rem 1.5rem' }}>3Y</th>
              <th style={{ background: '#fff', color: '#22223b', fontWeight: 700, fontSize: '1.3rem', padding: '1rem 1.5rem' }}>5Y</th>
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
                  <td style={{ color: '#22223b', fontWeight: 500, fontSize: '1.1rem', padding: '1rem 1.5rem' }}>
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
                  <td style={{ background: row.y1 > 0 ? '#d1fae5' : row.y1 < 0 ? '#fee2e2' : '#f3f4f6', color: row.y1 > 0 ? '#065f46' : row.y1 < 0 ? '#991b1b' : '#22223b', borderRadius: '0.75rem', padding: '0.5rem 1rem', fontWeight: 600 }}>{row.y1}%</td>
                  <td style={{ background: row.y3 > 0 ? '#d1fae5' : row.y3 < 0 ? '#fee2e2' : '#f3f4f6', color: row.y3 > 0 ? '#065f46' : row.y3 < 0 ? '#991b1b' : '#22223b', borderRadius: '0.75rem', padding: '0.5rem 1rem', fontWeight: 600 }}>{row.y3}%</td>
                  <td style={{ background: row.y5 > 0 ? '#d1fae5' : row.y5 < 0 ? '#fee2e2' : '#f3f4f6', color: row.y5 > 0 ? '#065f46' : row.y5 < 0 ? '#991b1b' : '#22223b', borderRadius: '0.75rem', padding: '0.5rem 1rem', fontWeight: 600 }}>{row.y5}%</td>
                </tr>
                {row.asset === 'US Stock Market (S&P 500)' && expanded && (
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
                      <td style={{ color: '#22223b', fontWeight: 400, fontSize: '1rem', padding: '0.75rem 1.5rem', paddingLeft: '2.5rem' }}>{sub.asset}</td>
                      <td style={{ background: sub.y1 > 0 ? '#d1fae5' : sub.y1 < 0 ? '#fee2e2' : '#f3f4f6', color: sub.y1 > 0 ? '#065f46' : sub.y1 < 0 ? '#991b1b' : '#22223b', borderRadius: '0.75rem', padding: '0.5rem 1rem', fontWeight: 500 }}>{sub.y1}%</td>
                      <td style={{ background: sub.y3 > 0 ? '#d1fae5' : sub.y3 < 0 ? '#fee2e2' : '#f3f4f6', color: sub.y3 > 0 ? '#065f46' : sub.y3 < 0 ? '#991b1b' : '#22223b', borderRadius: '0.75rem', padding: '0.5rem 1rem', fontWeight: 500 }}>{sub.y3}%</td>
                      <td style={{ background: sub.y5 > 0 ? '#d1fae5' : sub.y5 < 0 ? '#fee2e2' : '#f3f4f6', color: sub.y5 > 0 ? '#065f46' : sub.y5 < 0 ? '#991b1b' : '#22223b', borderRadius: '0.75rem', padding: '0.5rem 1rem', fontWeight: 500 }}>{sub.y5}%</td>
                    </tr>
                  ))
                )}
              </React.Fragment>
            ))}
          </tbody>
        </table>
      </div>
      <div style={{ textAlign: 'right', marginTop: '1.5rem' }}>
        <button
          className="btn-primary"
          disabled={!isAnySelected}
          onClick={isAnySelected ? () => onCreatePortfolio({ checked, subChecked }) : undefined}
          style={{
            background: !isAnySelected ? '#e5e7eb' : '#22223b',
            color: !isAnySelected ? '#888' : '#fff',
            border: 'none',
            borderRadius: '1rem',
            padding: '0.75rem 2rem',
            fontSize: '1.1rem',
            fontWeight: 600,
            boxShadow: '0 2px 8px rgba(30,41,59,0.10)',
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
            <th style={{ background: '#fff', color: '#22223b', fontWeight: 700, fontSize: '1.3rem', padding: '1rem 1.5rem' }}>S&P 500 Sector</th>
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
        <h2 style={{ fontSize: '2.2rem', fontWeight: 700, marginBottom: '1.5rem', color: '#22223b', fontFamily: 'Poppins, Inter, sans-serif' }}>Templates</h2>
        <p style={{ color: '#22223b', fontSize: '1.1rem', fontFamily: 'Poppins, Inter, sans-serif' }}>
          Here you can manage and select portfolio templates. (Demo content)
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

  // S&P 500 subsectors state lifted up
  const subsectors = [
    { asset: 'Communication Services', y1: 8.2, y3: 5.1, y5: 2.3 },
    { asset: 'Consumer Discretionary', y1: 7.5, y3: 6.2, y5: 3.1 },
    { asset: 'Consumer Staples', y1: 6.8, y3: 4.9, y5: 2.7 },
    { asset: 'Energy', y1: 5.9, y3: 3.8, y5: 1.9 },
    { asset: 'Financials', y1: 9.1, y3: 7.2, y5: 4.5 },
    { asset: 'Health Care', y1: 10.3, y3: 8.7, y5: 5.6 },
    { asset: 'Industrials', y1: 6.2, y3: 5.0, y5: 2.8 },
    { asset: 'Information Technology', y1: 12.4, y3: 10.1, y5: 7.3 },
    { asset: 'Materials', y1: 4.7, y3: 3.2, y5: 1.5 },
    { asset: 'Real Estate', y1: 3.5, y3: 2.1, y5: 0.9 },
    { asset: 'Utilities', y1: 2.8, y3: 1.7, y5: 0.6 },
  ];
  const [subChecked, setSubChecked] = React.useState(Array(subsectors.length).fill(true));
  const isStockMarketChecked = subChecked.every(Boolean);

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

  const handleCreatePortfolio = ({ checked, subChecked }) => {
    setLoading(true);
    setTimeout(() => {
      setLoading(false);
      setShowPortfolio(true);
      setActiveSegment(3); // Switch to Portfolios tab
      setPortfolioData({ checked, subChecked });
    }, 2000);
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
        <div className="card" style={{ display: 'flex', flexWrap: 'wrap', gap: '2rem', background: '#f3f4f6', borderRadius: '2rem', boxShadow: '0 2px 8px rgba(30,41,59,0.04)', minHeight: '50vh', alignItems: 'stretch' }}>
          <div style={{ flex: '1 1 0', minWidth: '300px', maxWidth: '100%', minHeight: '220px', display: 'flex', alignItems: 'center', justifyContent: 'center', width: '50%', boxSizing: 'border-box' }}>
            {/* Placeholder for chart */}
            <div style={{ width: '100%', height: '100%', background: '#f3f4f6', borderRadius: '1.25rem', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#8b5cf6', fontWeight: 600, fontSize: '1.1rem' }}>
              [Chart Area]
            </div>
          </div>
          <div style={{ flex: '1 1 0', minWidth: '300px', maxWidth: '100%', width: '50%', boxSizing: 'border-box', overflowX: 'visible' }}>
            <h2 style={{ fontSize: '2.2rem', fontWeight: 700, marginBottom: '1rem', color: '#22223b' }}>
              Asset Class Performance
            </h2>
            <AssetPerformanceTable
              data={assetPerformanceData}
              checked={checked}
              handleCheck={handleCheck}
              onCreatePortfolio={handleCreatePortfolio}
              subsectors={subsectors}
              subChecked={subChecked}
              setSubChecked={setSubChecked}
            />
          </div>
        </div>
      ) : activeSegment === 1 ? (
        // S&P 500 Stocks tab content (reuse your S&P 500 table and chart layout)
        <div className="card" style={{ display: 'flex', flexWrap: 'wrap', gap: '2rem', background: '#f3f4f6', borderRadius: '2rem', boxShadow: '0 2px 8px rgba(30,41,59,0.04)', minHeight: '50vh', alignItems: 'stretch', padding: '2rem' }}>
          <div style={{ flex: '1 1 0', minWidth: '300px', maxWidth: '100%', minHeight: '220px', display: 'flex', alignItems: 'center', justifyContent: 'center', width: '50%', boxSizing: 'border-box' }}>
            <div style={{ width: '100%', height: '100%', background: '#f3f4f6', borderRadius: '1.25rem', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#8b5cf6', fontWeight: 600, fontSize: '1.1rem' }}>
              [Chart Area]
            </div>
          </div>
          <div style={{ flex: '1 1 0', minWidth: '300px', maxWidth: '100%', width: '50%', boxSizing: 'border-box', overflowX: 'visible' }}>
            <h2 style={{ fontSize: '2.2rem', fontWeight: 700, marginBottom: '1rem', color: '#22223b' }}>
              S&P 500 Sectors
            </h2>
            <SP500StocksTable subsectors={subsectors} subChecked={subChecked} handleSubCheck={handleSubCheck} />
          </div>
        </div>
      ) : (
        <TemplatesPage />
      )}
      <style>{`
        @media (max-width: 900px) {
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
            font-size: 1.1rem !important;
          }
          .card {
            padding: 0.5rem !important;
            border-radius: 1rem !important;
          }
          .data-table th, .data-table td {
            padding: 0.5rem 0.5rem !important;
            font-size: 0.9rem !important;
          }
        }
      `}</style>
    </div>
  );
}

export default DashboardDemo;