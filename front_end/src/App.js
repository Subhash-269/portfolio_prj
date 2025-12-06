import React, { useState } from 'react';
import { PieChart, Pie, BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell, Area, AreaChart } from 'recharts';
import { TrendingUp, TrendingDown, DollarSign, Activity, Target, Sparkles, ChevronRight, Play, Download } from 'lucide-react';

const PortfolioVisualizer = () => {
  const [step, setStep] = useState('input');
  const [capital, setCapital] = useState('100000');
  const [error, setError] = useState('');
  const [selectedAssets, setSelectedAssets] = useState({
    stockMarket: true,
    commodities: false,
    bonds: false,
    crypto: false,
    forex: false
  });
  
  const [selectedSubAssets, setSelectedSubAssets] = useState({
    tech: true,
    health: true,
    finance: false,
    consumer: false,
    energy: false,
    gold: false,
    silver: false,
    copper: false,
    oil: false
  });

  const assetCategories = {
    stockMarket: {
      name: 'Stock Market',
      icon: '📈',
      subAssets: ['Tech', 'Health', 'Finance', 'Consumer', 'Energy']
    },
    commodities: {
      name: 'Commodities',
      icon: '🏆',
      subAssets: ['Gold', 'Silver', 'Copper', 'Oil']
    },
    bonds: {
      name: 'Bonds',
      icon: '🏦',
      subAssets: []
    },
    crypto: {
      name: 'Cryptocurrency',
      icon: '₿',
      subAssets: []
    },
    forex: {
      name: 'Foreign Currency',
      icon: '💱',
      subAssets: []
    }
  };

  const optimizedAllocation = [
    { ticker: 'AAPL', sector: 'Tech', allocation: 18.5, value: 18500, price: 178.50, change: 2.3 },
    { ticker: 'MSFT', sector: 'Tech', allocation: 16.2, value: 16200, price: 372.40, change: 1.8 },
    { ticker: 'GOOGL', sector: 'Tech', allocation: 14.8, value: 14800, price: 141.20, change: -0.5 },
    { ticker: 'JNJ', sector: 'Health', allocation: 15.5, value: 15500, price: 158.30, change: 0.8 },
    { ticker: 'UNH', sector: 'Health', allocation: 12.3, value: 12300, price: 524.80, change: 1.2 },
    { ticker: 'PFE', sector: 'Health', allocation: 11.2, value: 11200, price: 28.90, change: -1.1 },
    { ticker: 'NVDA', sector: 'Tech', allocation: 11.5, value: 11500, price: 495.20, change: 3.5 }
  ];

  const performanceData = [
    { date: 'Week 1', portfolio: 100000, benchmark: 100000, optimized: 100000 },
    { date: 'Week 2', portfolio: 101200, benchmark: 100800, optimized: 102100 },
    { date: 'Week 3', portfolio: 103500, benchmark: 101500, optimized: 104800 },
    { date: 'Week 4', portfolio: 105800, benchmark: 102300, optimized: 107200 },
    { date: 'Week 5', portfolio: 107500, benchmark: 103100, optimized: 109800 },
    { date: 'Week 6', portfolio: 109200, benchmark: 104000, optimized: 112500 },
  ];

  const metrics = {
    totalValue: 112500,
    totalReturn: 12.5,
    sharpeRatio: 2.14,
    volatility: 14.7,
    maxDrawdown: -8.2,
    alpha: 1.12,
    beta: 0.95
  };

  const COLORS = ['#6366f1', '#8b5cf6', '#ec4899', '#f59e0b', '#10b981', '#06b6d4', '#3b82f6'];

  const handleOptimize = () => {
    const capitalAmount = Number(capital);
    if (capitalAmount < 10000) {
      setError('Minimum investment capital is $10,000');
      return;
    }
    
    const hasSelectedAsset = Object.values(selectedAssets).some(v => v);
    if (!hasSelectedAsset) {
      setError('Please select at least one asset class');
      return;
    }
    
    setError('');
    setStep('output');
  };

  const toggleAsset = (asset) => {
    setSelectedAssets(prev => ({ ...prev, [asset]: !prev[asset] }));
  };

  const toggleSubAsset = (subAsset) => {
    setSelectedSubAssets(prev => ({ ...prev, [subAsset.toLowerCase()]: !prev[subAsset.toLowerCase()] }));
  };

  if (step === 'input') {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 p-8">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-12">
            <div className="inline-flex items-center justify-center w-16 h-16 bg-gradient-to-br from-purple-500 to-pink-500 rounded-2xl mb-4 shadow-lg shadow-purple-500/50">
              <Sparkles className="w-8 h-8 text-white" />
            </div>
            <h1 className="text-5xl font-bold text-white mb-3 bg-clip-text text-transparent bg-gradient-to-r from-purple-400 to-pink-400">
              AI Portfolio Optimizer
            </h1>
            <p className="text-xl text-purple-200">Maximize returns with reinforcement learning</p>
          </div>

          <div className="bg-white/10 backdrop-blur-xl rounded-3xl p-8 border border-white/20 shadow-2xl">
            <div className="mb-8">
              <label className="block text-white text-lg font-semibold mb-3 flex items-center">
                <DollarSign className="w-5 h-5 mr-2 text-green-400" />
                Investment Capital
              </label>
              <div className="relative">
                <span className="absolute left-6 top-1/2 -translate-y-1/2 text-2xl text-purple-300">$</span>
                <input
                  type="text"
                  value={capital}
                  onChange={(e) => {
                    setCapital(e.target.value.replace(/[^0-9]/g, ''));
                    setError('');
                  }}
                  className="w-full bg-white/5 border-2 border-purple-500/30 rounded-2xl pl-12 pr-6 py-4 text-2xl text-white placeholder-purple-300/50 focus:outline-none focus:border-purple-400 transition-all"
                  placeholder="100000"
                />
              </div>
              <p className="mt-2 text-purple-200 text-sm ml-2">
                {capital && `≈ ${Number(capital).toLocaleString()} USD`}
              </p>
              {error && (
                <div className="mt-3 bg-red-500/20 border border-red-500/50 rounded-xl p-3 flex items-center">
                  <div className="w-2 h-2 bg-red-500 rounded-full mr-2"></div>
                  <p className="text-red-200 text-sm">{error}</p>
                </div>
              )}
              <p className="mt-2 text-purple-300/60 text-xs ml-2">
                Minimum investment: $10,000
              </p>
            </div>

            <div>
              <label className="block text-white text-lg font-semibold mb-4 flex items-center">
                <Target className="w-5 h-5 mr-2 text-pink-400" />
                Asset Classes & Categories
              </label>
              
              <div className="space-y-3">
                {Object.entries(assetCategories).map(([key, { name, icon, subAssets }]) => (
                  <div key={key} className="bg-white/5 rounded-2xl border border-white/10 overflow-hidden transition-all hover:bg-white/10">
                    <div 
                      className="flex items-center p-4 cursor-pointer"
                      onClick={() => toggleAsset(key)}
                    >
                      <input
                        type="checkbox"
                        checked={selectedAssets[key]}
                        onChange={() => {}}
                        className="w-5 h-5 rounded border-2 border-purple-400 text-purple-600 focus:ring-0 focus:ring-offset-0 cursor-pointer"
                      />
                      <span className="text-2xl ml-4 mr-3">{icon}</span>
                      <span className="text-white font-medium text-lg flex-1">{name}</span>
                      {subAssets.length > 0 && (
                        <ChevronRight className={`w-5 h-5 text-purple-300 transition-transform ${selectedAssets[key] ? 'rotate-90' : ''}`} />
                      )}
                    </div>
                    
                    {selectedAssets[key] && subAssets.length > 0 && (
                      <div className="px-4 pb-4 pt-2 border-t border-white/10 bg-white/5">
                        <div className="grid grid-cols-2 md:grid-cols-3 gap-3 pl-8">
                          {subAssets.map(subAsset => (
                            <label key={subAsset} className="flex items-center cursor-pointer group">
                              <input
                                type="checkbox"
                                checked={selectedSubAssets[subAsset.toLowerCase()]}
                                onChange={() => toggleSubAsset(subAsset)}
                                className="w-4 h-4 rounded border-2 border-purple-400 text-purple-600 focus:ring-0 cursor-pointer"
                              />
                              <span className="ml-2 text-purple-100 text-sm group-hover:text-white transition-colors">
                                {subAsset}
                              </span>
                            </label>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>

            <button
              onClick={handleOptimize}
              className="w-full mt-8 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-500 hover:to-pink-500 text-white font-bold py-5 rounded-2xl transition-all transform hover:scale-[1.02] active:scale-[0.98] shadow-lg shadow-purple-500/50 flex items-center justify-center text-lg"
            >
              <Play className="w-6 h-6 mr-2" />
              Optimize Portfolio with AI
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 p-6">
      <div className="max-w-7xl mx-auto">
        <div className="flex items-center justify-between mb-8">
          <div>
            <button
              onClick={() => setStep('input')}
              className="text-purple-300 hover:text-white mb-3 flex items-center text-sm transition-colors"
            >
              ← Back to inputs
            </button>
            <h1 className="text-4xl font-bold text-white mb-2">Optimized Portfolio</h1>
            <p className="text-purple-200">AI-generated allocation for maximum risk-adjusted returns</p>
          </div>
          <button className="bg-white/10 hover:bg-white/20 backdrop-blur-xl border border-white/20 text-white px-6 py-3 rounded-xl flex items-center transition-all">
            <Download className="w-4 h-4 mr-2" />
            Export Report
          </button>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
          <div className="bg-gradient-to-br from-green-500/20 to-emerald-500/20 backdrop-blur-xl rounded-2xl p-5 border border-green-500/30 shadow-lg">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-green-100">Portfolio Value</span>
              <DollarSign className="w-4 h-4 text-green-400" />
            </div>
            <div className="flex items-end justify-between">
              <span className="text-3xl font-bold text-white">${(metrics.totalValue / 1000).toFixed(1)}K</span>
              <span className="text-sm flex items-center text-green-400 font-semibold">
                <TrendingUp className="w-4 h-4 mr-1" />
                +{metrics.totalReturn}%
              </span>
            </div>
          </div>

          <div className="bg-gradient-to-br from-purple-500/20 to-pink-500/20 backdrop-blur-xl rounded-2xl p-5 border border-purple-500/30 shadow-lg">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-purple-100">Sharpe Ratio</span>
              <Activity className="w-4 h-4 text-purple-400" />
            </div>
            <span className="text-3xl font-bold text-white">{metrics.sharpeRatio.toFixed(2)}</span>
            <p className="text-xs text-purple-200 mt-1">Excellent risk-adjusted return</p>
          </div>

          <div className="bg-gradient-to-br from-blue-500/20 to-cyan-500/20 backdrop-blur-xl rounded-2xl p-5 border border-blue-500/30 shadow-lg">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-blue-100">Volatility</span>
              <TrendingUp className="w-4 h-4 text-blue-400" />
            </div>
            <span className="text-3xl font-bold text-white">{metrics.volatility.toFixed(1)}%</span>
            <p className="text-xs text-blue-200 mt-1">Annualized standard deviation</p>
          </div>

          <div className="bg-gradient-to-br from-orange-500/20 to-red-500/20 backdrop-blur-xl rounded-2xl p-5 border border-orange-500/30 shadow-lg">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-orange-100">Max Drawdown</span>
              <TrendingDown className="w-4 h-4 text-orange-400" />
            </div>
            <span className="text-3xl font-bold text-white">{metrics.maxDrawdown.toFixed(1)}%</span>
            <p className="text-xs text-orange-200 mt-1">Peak-to-trough decline</p>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2 space-y-6">
            <div className="bg-white/10 backdrop-blur-xl rounded-3xl p-6 border border-white/20 shadow-2xl">
              <h2 className="text-xl font-semibold text-white mb-4 flex items-center">
                <div className="w-2 h-8 bg-gradient-to-b from-purple-500 to-pink-500 rounded-full mr-3"></div>
                Portfolio Performance
              </h2>
              <ResponsiveContainer width="100%" height={300}>
                <AreaChart data={performanceData}>
                  <defs>
                    <linearGradient id="colorPortfolio" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.8}/>
                      <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0}/>
                    </linearGradient>
                    <linearGradient id="colorBenchmark" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#64748b" stopOpacity={0.4}/>
                      <stop offset="95%" stopColor="#64748b" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#ffffff20" />
                  <XAxis dataKey="date" stroke="#a78bfa" />
                  <YAxis stroke="#a78bfa" />
                  <Tooltip contentStyle={{ backgroundColor: '#1e1b4b', border: '1px solid #8b5cf6', borderRadius: '12px' }} />
                  <Legend />
                  <Area type="monotone" dataKey="optimized" stroke="#8b5cf6" fillOpacity={1} fill="url(#colorPortfolio)" strokeWidth={3} name="AI Optimized" />
                  <Area type="monotone" dataKey="benchmark" stroke="#64748b" fillOpacity={1} fill="url(#colorBenchmark)" strokeWidth={2} name="S&P 500" />
                </AreaChart>
              </ResponsiveContainer>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="bg-white/10 backdrop-blur-xl rounded-3xl p-6 border border-white/20 shadow-2xl">
                <h2 className="text-xl font-semibold text-white mb-4">Allocation Distribution</h2>
                <ResponsiveContainer width="100%" height={250}>
                  <PieChart>
                    <Pie
                      data={optimizedAllocation}
                      cx="50%"
                      cy="50%"
                      innerRadius={60}
                      outerRadius={90}
                      paddingAngle={2}
                      dataKey="allocation"
                    >
                      {optimizedAllocation.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip contentStyle={{ backgroundColor: '#1e1b4b', border: '1px solid #8b5cf6', borderRadius: '12px' }} />
                  </PieChart>
                </ResponsiveContainer>
              </div>

              <div className="bg-white/10 backdrop-blur-xl rounded-3xl p-6 border border-white/20 shadow-2xl">
                <h2 className="text-xl font-semibold text-white mb-4">Sector Allocation</h2>
                <ResponsiveContainer width="100%" height={250}>
                  <BarChart data={optimizedAllocation}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#ffffff20" />
                    <XAxis dataKey="ticker" stroke="#a78bfa" />
                    <YAxis stroke="#a78bfa" />
                    <Tooltip contentStyle={{ backgroundColor: '#1e1b4b', border: '1px solid #8b5cf6', borderRadius: '12px' }} />
                    <Bar dataKey="allocation" fill="#8b5cf6" radius={[8, 8, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="bg-white/10 backdrop-blur-xl rounded-3xl p-6 border border-white/20 shadow-2xl">
              <h2 className="text-xl font-semibold text-white mb-4">Optimized Holdings</h2>
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-white/20">
                      <th className="text-left py-3 px-4 text-sm font-semibold text-purple-200">Ticker</th>
                      <th className="text-left py-3 px-4 text-sm font-semibold text-purple-200">Sector</th>
                      <th className="text-right py-3 px-4 text-sm font-semibold text-purple-200">Price</th>
                      <th className="text-right py-3 px-4 text-sm font-semibold text-purple-200">Allocation</th>
                      <th className="text-right py-3 px-4 text-sm font-semibold text-purple-200">Value</th>
                      <th className="text-right py-3 px-4 text-sm font-semibold text-purple-200">Change</th>
                    </tr>
                  </thead>
                  <tbody>
                    {optimizedAllocation.map((holding, idx) => (
                      <tr key={idx} className="border-b border-white/10 hover:bg-white/5 transition-colors">
                        <td className="py-3 px-4 font-bold text-white">{holding.ticker}</td>
                        <td className="py-3 px-4 text-purple-200">{holding.sector}</td>
                        <td className="py-3 px-4 text-right text-purple-100">${holding.price.toFixed(2)}</td>
                        <td className="py-3 px-4 text-right text-white font-semibold">{holding.allocation.toFixed(1)}%</td>
                        <td className="py-3 px-4 text-right text-purple-100">${holding.value.toLocaleString()}</td>
                        <td className={`py-3 px-4 text-right font-semibold ${
                          holding.change >= 0 ? 'text-green-400' : 'text-red-400'
                        }`}>
                          {holding.change >= 0 ? '+' : ''}{holding.change}%
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          <div className="space-y-6">
            <div className="bg-white/10 backdrop-blur-xl rounded-3xl p-6 border border-white/20 shadow-2xl">
              <h3 className="text-lg font-semibold text-white mb-4">Risk Analysis</h3>
              <div className="space-y-4">
                <div>
                  <div className="flex justify-between mb-2">
                    <span className="text-sm text-purple-200">Alpha</span>
                    <span className="text-sm font-bold text-white">{metrics.alpha.toFixed(2)}</span>
                  </div>
                  <div className="w-full bg-white/10 rounded-full h-2">
                    <div className="bg-gradient-to-r from-green-500 to-emerald-400 h-2 rounded-full" style={{ width: `${(metrics.alpha / 2) * 100}%` }}></div>
                  </div>
                </div>
                <div>
                  <div className="flex justify-between mb-2">
                    <span className="text-sm text-purple-200">Beta</span>
                    <span className="text-sm font-bold text-white">{metrics.beta.toFixed(2)}</span>
                  </div>
                  <div className="w-full bg-white/10 rounded-full h-2">
                    <div className="bg-gradient-to-r from-blue-500 to-cyan-400 h-2 rounded-full" style={{ width: `${(metrics.beta / 2) * 100}%` }}></div>
                  </div>
                </div>
                <div>
                  <div className="flex justify-between mb-2">
                    <span className="text-sm text-purple-200">Sharpe Ratio</span>
                    <span className="text-sm font-bold text-white">{metrics.sharpeRatio.toFixed(2)}</span>
                  </div>
                  <div className="w-full bg-white/10 rounded-full h-2">
                    <div className="bg-gradient-to-r from-purple-500 to-pink-400 h-2 rounded-full" style={{ width: `${(metrics.sharpeRatio / 3) * 100}%` }}></div>
                  </div>
                </div>
              </div>
            </div>

            <div className="bg-gradient-to-br from-purple-500/20 to-pink-500/20 backdrop-blur-xl rounded-3xl p-6 border border-purple-500/30 shadow-2xl">
              <div className="flex items-center mb-3">
                <Sparkles className="w-5 h-5 text-purple-300 mr-2" />
                <h3 className="text-lg font-semibold text-white">AI Methodology</h3>
              </div>
              <p className="text-sm text-purple-100 mb-4">
                Portfolio optimized using deep reinforcement learning with policy gradient methods for maximum Sharpe ratio.
              </p>
              <div className="space-y-2">
                <div className="flex items-center text-sm text-purple-100">
                  <div className="w-2 h-2 bg-purple-400 rounded-full mr-2"></div>
                  Policy Gradient (PG) Algorithm
                </div>
                <div className="flex items-center text-sm text-purple-100">
                  <div className="w-2 h-2 bg-pink-400 rounded-full mr-2"></div>
                  Real-time market adaptation
                </div>
                <div className="flex items-center text-sm text-purple-100">
                  <div className="w-2 h-2 bg-indigo-400 rounded-full mr-2"></div>
                  Risk-adjusted optimization
                </div>
              </div>
            </div>

            <div className="bg-white/10 backdrop-blur-xl rounded-3xl p-6 border border-white/20 shadow-2xl">
              <h3 className="text-lg font-semibold text-white mb-4">Portfolio Summary</h3>
              <div className="space-y-3">
                <div className="flex justify-between">
                  <span className="text-purple-200 text-sm">Initial Capital</span>
                  <span className="text-white font-semibold">${Number(capital).toLocaleString()}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-purple-200 text-sm">Current Value</span>
                  <span className="text-white font-semibold">${metrics.totalValue.toLocaleString()}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-purple-200 text-sm">Total Gain</span>
                  <span className="text-green-400 font-semibold">+${(metrics.totalValue - Number(capital)).toLocaleString()}</span>
                </div>
                <div className="flex justify-between pt-3 border-t border-white/20">
                  <span className="text-purple-200 text-sm">Number of Assets</span>
                  <span className="text-white font-semibold">{optimizedAllocation.length}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-purple-200 text-sm">Last Optimized</span>
                  <span className="text-white font-semibold text-xs">
                    {new Date().toLocaleDateString()}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default PortfolioVisualizer;